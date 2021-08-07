import torch

# ===== Parameters

# Do not modify
HEIGHT = 40960
# Set this to 256 to get Stockfish's model
WIDTH = 128

# The ReLU parameters of the model
RELU_MIN = 0
RELU_MAX = 1

# ===== Parsing

# Parse a single line of the format FEN;EVAL
def parse_sample(line):
    fen, eval_w = line.split(";")

    # From centipawns to pawns
    eval_w /= 100

    white_indexes = {
        "P": 0, "N": 128, "B": 256, 
        "R": 384, "Q": 512, "K": 640,
    }

    black_indexes = {
        "p": 0, "n": 128, "b": 256, 
        "r": 384, "q": 512, "k": 640,
    }

    indexes_w = []
    indexes_b = []

    sq = 56
    for rank in fen.split("/"):
        for char in rank:
            if index := white_indexes.get(char):
                # It's a white piece
                indexes_w.append((index, sq))
                if char == "K":
                    king_w = 640 * sq
                sq += 1
            elif index == black_indexes.get(char):
                # It's a black piece
                indexes_b.append((index, sq))
                if char == "k":
                    king_b = 640 * (sq ^ 63)
                sq += 1
            else:
                # It's a number
                sq += ord(char) - ord("0")
        sq -= 16

    sample_w = []
    sample_b = []

    for index, sq in indexes_w:
        sample_w.append(king_w + index + sq)
        sample_b.append(king_b + index + 64 + (sq ^ 63))
    for index, sq in indexes_b:
        sample_w.append(king_w + index + 64 + sq)
        sample_b.append(king_b + index + (sq ^ 63))

    sample_w.sort()
	sample_b.sort()

    eval_b = -eval_w

    return (sample_w, sample_b, eval_w, eval_b)

def load_batch(file):
    pass # TODO

# ===== Model

# Straight from https://github.com/david-carteau/cerebrum.
class Net(torch.nn.Model()):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(HEIGHT, WIDTH)
		self.l2 = torch.nn.Linear(WIDTH * 2, 32)
		self.l3 = torch.nn.Linear(32, 32)
		self.l4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x1, x2 = x

        x1 = self.l1(x1)
        x2 = self.l2(x2)
        x = torch.cat((x1, x2), axis=1)
        x = torch.clamp(x, min=RELU_MIN, max=RELU_MAX)
        
        x = self.l2(x)
        x = torch.clamp(x, min=RELU_MIN, max=RELU_MAX)

        x = self.l3(x)
        x = torch.clamp(x, min=RELU_MIN, max=RELU_MAX)
        
        return self.l4(x)

# ===== main()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Net().to(device)
    print(model)

if __name__ == "__main__":
    main()
