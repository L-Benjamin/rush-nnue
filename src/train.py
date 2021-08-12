import argparse
import bz2
import json
import os
import random
import time

import torch as tch

# ===== Parameters

# 64 piece's squares x 64 king's square x 5 non-king piece types x 2 colors.
HEIGHT = 40960

# Set this to 256 to get Stockfish's old HalfKP model. We want something leaner.
SIZE = 128

# The loss function used.
LOSS_FN = tch.nn.MSELoss

# The optimizer used for gradient descent.
OPTIMIZER = tch.optim.Adagrad

# ===== Parsing

def parse_sample(line):
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

    fen, centipawns = line.split(";")

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

    eval_w = int(centipawns)
    eval_b = -eval_w

    return sample_w, sample_b, eval_w, eval_b

def load_batch(file_name):
    """
    Loads a batch file from it's name, decompresses it and parses it. 

    Returns three tensors, two containing the inputs of the net, for both
    colors, and one containing the labels.
    """

    with bz2.open(file_name, "rt") as f:
        lines = f.readlines()
    
    random.shuffle(lines)

    # The x and y indices of the places where we need
    # a "1" in the input matrixes.
    ix1, iy1 = [], []
    ix2, iy2 = [], []

    # The label's vector.
    y = tch.empty((len(lines) * 2, 1), dtype=tch.int16)

    # Parse each line of the batch file
    i = 0
    for line in lines:
        sample_w, sample_b, eval_w, eval_b = parse_sample(line)

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
    X1 = tch.sparse_coo_tensor([iy1, ix1], tch.ones(len(iy1)), size=(len(lines) * 2, HEIGHT), dtype=tch.int16)
    X2 = tch.sparse_coo_tensor([iy2, ix2], tch.ones(len(iy2)), size=(len(lines) * 2, HEIGHT), dtype=tch.int16)

    return (X1, X2), y

# ===== Files

def verify_dir(dirname):
    """
    Exits the program with an error code if the argument is not 
    a path to a directory.
    """

    if not os.path.isdir(dirname):
        print(f"{dirname} is not a directory.")
        exit(1)

def load_net(device, filename):
    """
    Loads or initializes a network from a given file name.
    Also chooses the best device to train the network on.
    Returns the device and the initialized model.
    """
    
    model = Net().to(device)

    if filename is not None:
        try:
            model.load_state_dict(tch.load(filename))
        except FileNotFoundError:
            print(f"Network file not found: {filename}")
            exit(1)

    return model

# ===== Network model

class Net(tch.nn.Module):
    """
    The main model for our network, based on a slighlty 
    modified version of Stockfish's HalfKP model.
    See: https://www.chessprogramming.org/Stockfish_NNUE#HalfKP.
    """

    def __init__(self):
        super().__init__()

        self.l1 = tch.nn.Linear(HEIGHT, SIZE)
        self.l2 = tch.nn.Linear(SIZE * 2, 32)
        self.l3 = tch.nn.Linear(32, 32)
        self.l4 = tch.nn.Linear(32, 1)

    def forward(self, x):
        x1, x2 = x

        x1 = self.input_layer(x1)
        x2 = self.input_layer(x2)
        x = tch.cat((x1, x2), axis=1)
        x = tch.clamp(x, min=0, max=1)
        
        x = self.l2(x)
        x = tch.clamp(x, min=0, max=1)

        x = self.l3(x)
        x = tch.clamp(x, min=0, max=1)

        return self.l4(x)

# ===== train, test and main functions

def train(args):
    """
    Trains the network for the specified number of epochs and on the
    given dataset.
    """

    verify_dir(args.out)
    verify_dir(args.data)

    data_files = os.listdir(args.data)

    device = "cuda" if tch.cuda.is_available() else "cpu"
    model = load_net(device, args.net)
    print(f"Using device: {device}\n")

    optimizer = OPTIMIZER(model.parameters(), lr=args.learning_rate)
    loss_fn = LOSS_FN()

    for i in range(args.epochs):
        print(f"==== Started epoch n°{i + 1} ====")

        avg_loss = 0

        for data_file in data_files:
            X, y = load_batch(os.path.join(args.data, data_file))

            yhat = model(X)
            loss = loss_fn(yhat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            avg_loss += loss
            print(f"loss: {loss:.3f}")

        avg_loss /= len(data_files)

        print(f"\naverage loss in epoch: {avg_loss:.3f}\n===== Ended epoch n°{i + 1} =====\n")
        tch.save(model.state_dict(), os.path.join(args.out, f"nnue-{hex(random.randint(0, 0xffffffff))}.pt"))

def to_json(args):
    """
    Converts a neural network file (.pt) into it's json representation.
    """

    names = ["w0", "b0", "w1", "b1", "w2", "b2", "w3", "b3"]
    params = load_net("cpu", args.net).parameters()

    json.dump(
        {name: param.data.tolist() for name, param in zip(names, params)}, 
        open(args.json, "w"),
    )

def cuda(args):
    """
    Prints a message corresponding to the availability of cuda.
    """

    if tch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

def test(args):
    """
    Tests the network on the given batch files.
    """

    verify_dir(args.data)

    data_files = os.listdir(args.data)

    device = "cuda" if tch.cuda.is_available() else "cpu"
    model = load_net(device, args.net)
    print(f"Using device: {device}\n")

    loss_fn = LOSS_FN()

    avg_loss = 0

    for data_file in data_files:
        X, y = load_batch(os.path.join(args.data, data_file))

        yhat = model(X)
        loss = loss_fn(yhat, y)

        loss = loss.item()
        avg_loss += loss
        print(f"loss: {loss:.3f}")

    avg_loss /= len(data_files)

    print(f"\naverage loss on batches: {avg_loss:.3f})

def main():
    """
    Parses the arguments passed to the script. Calls either train() or test() 
    with the parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Train or test the nnue on batch files generated by the make-dataset script")
    subparsers = parser.add_subparsers(
        title="subcommands",
        required=True,
    )

    # "train" subcommand.
    train_subparser = subparsers.add_parser(
        name="train",
        description="Train the nnue",
    )
    train_subparser.add_argument(
        "data",
        metavar="DATA",
        help="The directory where the training data is stored",
    )
    train_subparser.add_argument(
        "--num-epochs", "-e",
        metavar="EPOCHS",
        help="The number of epochs to train the network for, defaults to 10",
        dest="epochs",
        type=int,
        default=10,
    )
    train_subparser.add_argument(
        "--learning-rate", "-l",
        metavar="RATE",
        help="The learning rate to use for training, defaults to 1e-3",
        dest="learning_rate",
        type=float,
        default=1e-3,
    )
    train_subparser.add_argument(
        "--net", "-n",
        metavar="NET",
        help="The network file that is loaded and used as starting point, randomly initializes the weights if this argument is missing",
        dest="net",
        default=None,
    )
    train_subparser.add_argument(
        "--out", "-o",
        metavar="OUT",
        help="The output directory to write nets to, defaults to the working directory",
        dest="out",
        default=".",
    )
    train_subparser.set_defaults(func=train)

    # "test" subcommand.
    test_subparser = subparsers.add_parser(
        name="test",
        description="Test the NNUE",
    )
    train_subparser.add_argument(
        "data",
        metavar="DATA",
        help="The directory where the test data is stored",
    )
    train_subparser.add_argument(
        "net",
        metavar="NET",
        help="The network file that is loaded and used as starting point, randomly initializes the weights if this argument is missing",
        dest="net",
        default=None,
    )
    test_subparser.set_defaults(func=test)

    # "json" subcommand
    json_subparser = subparsers.add_parser(
        name="json",
        description="Converts a nnue file (.pt) to the json format",
    )
    json_subparser.add_argument(
        "net",
        metavar="NET",
        help="The network file that is to be converted",
    )
    json_subparser.add_argument(
        "json",
        metavar="JSON",
        help="The path of the output json file",
    )
    json_subparser.set_defaults(func=to_json)

    # "cuda" subcommand
    cuda_subparser = subparsers.add_parser(
        name="cuda",
        description="Test for availability of cuda drivers",
    )
    cuda_subparser.set_defaults(func=cuda)

    try:
        args = parser.parse_args()
    except TypeError: # Yay, python
        parser.parse_args(["-h"])
        exit(1)

    args.func(args)
    print("Done!")

if __name__ == "__main__":
    main()