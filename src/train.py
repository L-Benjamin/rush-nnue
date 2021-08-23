import argparse
import bz2
import json
import os
import pickle
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

# ===== Files

def check_is_dir(dirname: str):
    """
    Exits the program with an error code if the argument is not 
    a path to a directory.
    """
    if not os.path.isdir(dirname):
        print(f"{dirname} is not a directory.")
        exit(1)

def load_net(device: str, filename: str):
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

def get_data_files(data_dir: str):
    """
    Returns the list of the names of the data files.
    """
    check_is_dir(data_dir)

    files = [f for f in os.listdir(data_dir)]
    if len(files) == 0:
        print("No data files found")
        exit(1)

    return files

def load_batch(device: str, file_name: str):
    """
    Loads a batch file from it's name and decompresses i.

    Returns three tensors, two containing the inputs of the net, for both
    colors, and one containing the labels.
    """
    if os.path.exists(file_name):
        with bz2.open(file_name, "r") as f:
            (X1, X2), y = pickle.load(f)
            return [(X1.to(device), X2.to(device)), y.to(device)]  

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

        x1 = self.l1(x1)
        x2 = self.l1(x2)
        x = tch.cat((x1, x2), axis=1)
        x = tch.clamp(x, min=0, max=1)
        
        x = self.l2(x)
        x = tch.clamp(x, min=0, max=1)

        x = self.l3(x)
        x = tch.clamp(x, min=0, max=1)

        return self.l4(x)

# ===== commands and main functions

def cmd_train(args):
    """
    Trains the network for the specified number of epochs and on the
    given dataset.
    """

    check_is_dir(args.out)
    data_files = get_data_files(args.data)

    device = "cuda" if tch.cuda.is_available() else "cpu"
    model = load_net(device, args.net)
    print(f"Using device: {device}\n")

    optimizer = OPTIMIZER(model.parameters(), lr=args.learning_rate)
    loss_fn = LOSS_FN()

    for e in range(args.epochs):
        print(f"==== Started epoch n°{e + 1} ====")

        random.shuffle(data_files)
        avg_loss = 0

        for data_file in data_files:
            X, y = load_batch(device, os.path.join(args.data, data_file))

            yhat = model(X)
            loss = loss_fn(yhat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            avg_loss += loss
            print(f"({data_file}) loss: {loss:.3f}")

        avg_loss /= len(data_files)

        print(f"\naverage loss in epoch: {avg_loss:.3f}\n===== Ended epoch n°{e + 1} =====\n")
        file_name = f"nnue-{random.randint(0, 0xffffffffff):010x}.pt"
        tch.save(model.state_dict(), os.path.join(args.out, file_name))

    print(f"Last network file saved: {file_name}")

def cmd_json(args):
    """
    Converts a neural network file (.pt) into it's json representation.
    """

    names = ["w0", "b0", "w1", "b1", "w2", "b2", "w3", "b3"]
    params = load_net("cpu", args.net).parameters()

    json.dump(
        {name: param.data.tolist() for name, param in zip(names, params)}, 
        open(args.json, "w"),
    )

def cmd_cuda(args):
    """
    Prints a message corresponding to the availability of cuda.
    """

    if tch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

def cmd_test(args):
    """
    Tests the network on the given batch files.
    """

    data_files = get_data_files(args.data)

    device = "cuda" if tch.cuda.is_available() else "cpu"
    model = load_net(device, args.net)
    print(f"Using device: {device}\n")

    loss_fn = LOSS_FN()

    avg_loss = 0

    for data_file in data_files:
        X, y = load_batch(device, os.path.join(args.data, data_file))

        yhat = model(X)
        loss = loss_fn(yhat, y)

        loss = loss.item()
        avg_loss += loss
        print(f"({data_file}) loss: {loss:.3f}")

    avg_loss /= len(data_files)

    print(f"\naverage loss on {len(data_files)} batches: {avg_loss:.3f}")

def cmd_eval(args):
    """
    Evaluates the network on a given fen string
    """

    model = load_net("cpu", args.net)

    ix1, ix2 = parse_sample(args.fen)
    iy1 = [0] * len(ix1)
    iy2 = [0] * len(ix2)

    x1 = tch.sparse_coo_tensor([iy1, ix1], tch.ones(len(ix1)), size=(1, HEIGHT))
    x2 = tch.sparse_coo_tensor([iy2, ix2], tch.ones(len(ix2)), size=(1, HEIGHT))

    y_w = model((x1, x2)).item()
    y_b = model((x2, x1)).item()

    print(f"The model gave an evaluation of:\n>>> {y_w:.3f} from white's point of view.\n>>> {y_b:.3f} from black's point of view.")

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
    train_subparser.set_defaults(func=cmd_train)

    # "test" subcommand.
    test_subparser = subparsers.add_parser(
        name="test",
        description="Test the NNUE",
    )
    test_subparser.add_argument(
        "data",
        metavar="DATA",
        help="The directory where the test data is stored",
    )
    test_subparser.add_argument(
        "net",
        metavar="NET",
        help="The network file that is loaded and used as starting point, randomly initializes the weights if this argument is missing",
    )
    test_subparser.set_defaults(func=cmd_test)

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
    json_subparser.set_defaults(func=cmd_json)

    # "cuda" subcommand
    cuda_subparser = subparsers.add_parser(
        name="cuda",
        description="Test for availability of cuda drivers",
    )
    cuda_subparser.set_defaults(func=cmd_cuda)

    # "eval" subcommand
    eval_subparser = subparsers.add_parser(
        name="eval",
        description="Evaluates a network on a given fen string."
    )
    eval_subparser.add_argument(
        "net",
        metavar="NET",
        help="The network file used for evaluation",
    )
    eval_subparser.add_argument(
        "fen",
        metavar="FEN",
        help="The fen string to be evaluated",
    )
    eval_subparser.set_defaults(func=cmd_eval)

    try:
        args = parser.parse_args()
    except TypeError: # Yay, python
        parser.parse_args(["-h"])
        exit(1)

    args.func(args)
    print("Done!")

if __name__ == "__main__":
    main()
