import bz2

import torch as tch

# ===== Parameters

# 64 piece's squares x 64 king's square x 5 non-king piece type x 2 colors.
HEIGHT = 40960

# Set this to 256 to get Stockfish's old model.
SIZE = 128

# The ReLU parameters of the model.
RELU_MIN = 0
RELU_MAX = 1

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

    eval_w = int(centipawns) / 100
    eval_b = -eval_w

    return sample_w, sample_b, eval_w, eval_b

def load_batch(file_name):
    """
    Loads a batch file from it's name, decompress it and parses it. 

    Returns three tensors, two containing the inputs of the net, for both
    colors, and one containing the labels.
    """

    with bz2.open(file_name, "rt") as f:
        lines = f.readlines()

    # The x and y indices of the places where we need
    # a "1" in the input matrixes.
    ix1, iy1 = [], []
    ix2, iy2 = [], []

    # The label's vector.
    Y = tch.empty(len(lines) * 2, dtype=tch.float)

    # Parse each line of the batch file
    i = 0
    for line in lines:
        sample_w, sample_b, eval_w, eval_b = parse_sample(line)

        # From white's point of view.
        ix1.extend(sample_w)
        iy1.extend([i] * len(sample_w))
        ix2.extend(sample_b)
        iy2.extend([i] * len(sample_b))
        Y[i] = eval_w
        i += 1

        # From black's point of view.
        ix1.extend(sample_b)
        iy1.extend([i] * len(sample_b))
        ix2.extend(sample_w)
        iy2.extend([i] * len(sample_w))
        Y[i] = eval_b
        i += 1

    # The inputs sparse tensors.
    X1 = tch.sparse_coo_tensor([iy1, ix1], tch.ones(len(iy1)), size=(len(lines) * 2, HEIGHT), dtype=tch.float)
    X2 = tch.sparse_coo_tensor([iy2, ix2], tch.ones(len(iy2)), size=(len(lines) * 2, HEIGHT), dtype=tch.float)

    return (X1, X2), Y

# ===== Model

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

        x1 = self.l1(x1)
        x2 = self.l1(x2)
        x = tch.cat((x1, x2), axis=1)
        x = tch.clamp(x, min=RELU_MIN, max=RELU_MAX)
        
        x = self.l2(x)
        x = tch.clamp(x, min=RELU_MIN, max=RELU_MAX)

        x = self.l3(x)
        x = tch.clamp(x, min=RELU_MIN, max=RELU_MAX)
        
        return self.l4(x)

# ===== main()

def main():
    """
    TODO
    """
    
    device = "cuda" if tch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Net().to(device)
    print(model)

if __name__ == "__main__":
    main()