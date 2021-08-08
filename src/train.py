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
            elif char == "K":
                # It's the white king.
                king_sq = 640 * sq
                sq += 1
            elif char == "k":
                # It's the black king.
                sq += 1
            else:
                # If it's a number.
                sq += ord(char) - ord("0")
        sq -= 16

    ix1 = []
    ix2 = []
    for piece_index, sq in indexes_w:
        ix1.append(king_sq + piece_index + sq)
    for piece_index, sq in indexes_b:
        ix2.append(king_sq + piece_index + sq + 64)

    y = int(centipawns) / 100

    return ix1, ix2, y

def load_batch(file_name):
    """
    Loads a batch file from it's name, decompress it and parses it. 

    Returns three tensors, two containing the inputs of the net, for both
    colors, and one containing labels.
    """

    with bz2.open(file_name, "rt") as f:
        lines = f.readlines()

    # The x and y indices of the places where we need
    # a "1" in the input matrixes.
    ix1, iy1 = [], []
    ix2, iy2 = [], []

    # The label's vector.
    Y = tch.empty(len(lines), dtype=tch.float)

    # Parse each line of the batch file
    for i, line in enumerate(lines):
        sample_ix1, sample_ix2, y = parse_sample(line)

        ix1.extend(sample_ix1)
        iy1.extend([i] * len(sample_ix1))

        ix2.extend(sample_ix2)
        iy2.extend([i] * len(sample_ix2))

        Y[i] = y

    # The inputs sparse tensors.
    X1 = tch.sparse_coo_tensor([iy1, ix1], tch.ones(len(ix1)), size=(len(lines), HEIGHT), dtype=tch.float)
    X2 = tch.sparse_coo_tensor([iy2, ix2], tch.ones(len(ix2)), size=(len(lines), HEIGHT), dtype=tch.float)

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
