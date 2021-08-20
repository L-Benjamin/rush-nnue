import argparse
import bz2
from multiprocessing import cpu_count, Process, Queue
import os
import pickle
import queue

import torch as tch

# 64 piece's squares x 64 king's square x 5 non-king piece types x 2 colors.
HEIGHT = 40960

# The size of the queue.
MAX_QUEUE = 8

def parse_sample(fen: str):
    """
    Parses a single line of the form "FEN;EVAL" where
    FEN is the FEN representation of a valid chess board, and
    EVAL is a signed integer, representing the evaluation of
    the board by an engine, in centipawns.

    Returns two lists and the evaluation of the position. The lists
    correspond to the index of the pieces for white and black.

    See: https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor.
    """
    
    WHITE_INDEXES = {"P": 0, "N": 128, "B": 256, "R": 384, "Q": 512}
    BLACK_INDEXES = {"p": 0, "n": 128, "b": 256, "r": 384, "q": 512}
    KINGS = {"K", "k"}

    indexes_w = []
    indexes_b = []

    sq = 56
    for rank in fen.split("/"):
        for char in rank:
            if (index := WHITE_INDEXES.get(char)) is not None:
                # If it's a white piece.
                indexes_w.append((index, sq))                
                sq += 1
            elif (index := BLACK_INDEXES.get(char)) is not None:
                # If it's a black piece.
                indexes_b.append((index, sq))
                sq += 1
            elif char in KINGS:
                # It's a king.
                if char == "K":
                    king_w = 640 * sq
                else:
                    king_b = 640 * (sq ^ 56)
                sq += 1
            else:
                # It's a number.
                sq += ord(char) - ord("0")
        sq -= 16

    sample_w = []
    sample_b = []

    for index, sq in indexes_w:
        sample_w.append(king_w + index + sq)
        sample_b.append(king_b + index + 64 + (sq ^ 56))
    for index, sq in indexes_b:
        sample_w.append(king_w + index + 64 + sq)
        sample_b.append(king_b + index + (sq ^ 56))

    return sample_w, sample_b

def parse_batch(in_queue: Queue):
    """
    Loads a batch file from it's name, decompresses it and parses it. 
    Caches the resulting tensors for later retrieval, and uses the cached
    results in priority.

    Returns three tensors, two containing the inputs of the net, for both
    colors, and one containing the labels.
    """
    while True:
        try:
            infile, outfile = in_queue.get(timeout=1)
        except queue.Empty:
            return

        with bz2.open(infile, "rt") as f:
            lines = f.readlines()

        # The x and y indices of the places where we need
        # a "1" in the input matrixes.
        ix1, iy1 = [], []
        ix2, iy2 = [], []

        # The label's vector.
        y = tch.empty((len(lines) * 2, 1))

        # Parse each line of the batch file
        i = 0
        for line in lines:
            fen, centipawns = line.split(";")

            sample_w, sample_b = parse_sample(fen)
            eval_w = float(centipawns) * 0.01
            eval_b = -eval_w

            # From white's point of view.
            ix1.extend(sample_w)
            iy1.extend([i] * len(sample_w))
            ix2.extend(sample_b)
            iy2.extend([i] * len(sample_b))
            y[i] = eval_w
            i += 1

            # From black's point of view.
            ix1.extend(sample_b)
            iy1.extend([i] * len(sample_b))
            ix2.extend(sample_w)
            iy2.extend([i] * len(sample_w))
            y[i] = eval_b
            i += 1

        # The inputs sparse tensors.
        X1 = tch.sparse_coo_tensor([iy1, ix1], tch.ones(len(iy1)), size=(len(lines) * 2, HEIGHT))
        X2 = tch.sparse_coo_tensor([iy2, ix2], tch.ones(len(iy2)), size=(len(lines) * 2, HEIGHT))

        res = [(X1, X2), y]

        with bz2.open(outfile, "w") as f:
            pickle.dump(res, f)

def split(prop: float, batches: str, train: str, test: str, out_queue: Queue):
    """
    Splits the batches dir into two after parsing them.
    """

    files = os.listdir(batches)
    files.sort()

    n = int(len(files) * prop)
    i = 0

    def put(f: str, dirname: str):
        nonlocal i
        out_queue.put((os.path.join(batches, f), os.path.join(dirname, f.replace("txt", "bin"))))
        i += 1
        print(f"Done: {i}/{len(files)}", end="\r")

    for f in files[:n]:
        put(f, train)
    for f in files[n:]:
        put(f, test)

    print()

def main():
    parser = argparse.ArgumentParser(description="Split the dataset into two subfolders (for training and testing), parsing the data into useable tensors for pytorch.")
    parser.add_argument(
        "batches",
        help="The input directory containig the batches",
        metavar="INPUT",
    )
    parser.add_argument(
        "train",
        help="The output directory containig the batches used for training",
        metavar="TRAIN",
    )
    parser.add_argument(
        "test",
        help="The output directory containig the batches used for testing",
        metavar="TRAIN",
    )
    parser.add_argument(
        "--proportion", "-p",
        help="The proportion of training batches, a percentage, defaults to 75",
        metavar="PERCENT",
        default=75,
        dest="prop",
        type=int,
    )

    args = parser.parse_args()
   
    queue = Queue(maxsize=MAX_QUEUE)

    for _ in range(cpu_count()):
        Process(
            target=parse_batch,
            kwargs={
                "in_queue": queue,
            }
        ).start()

    split(
        args.prop / 100,
        args.batches,
        args.train,
        args.test,
        queue,
    )    

if __name__ == "__main__":
    main()
