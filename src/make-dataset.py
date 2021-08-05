import argparse
import sys

import chess, chess.pgn

# Reads PGN games from stdin and converts them into evaluated FEN positions.
def main():
    # The chess board.
    board = chess.Board()

    # Keeps track of the number of position extracted.
    pos_count = 0
    bytes_count = 0

    # For each game provided through stdin.
    while game := chess.pgn.read_game(sys.stdin):
        # Reset the board to it's starting position.
        board.reset()

        # For each position in the game.
        while game := game.next():
            # Get the evaluation on this node. Not all game are evaluated sadly.
            if not (evaluation := game.eval()):
                break

            # Push that move to the board.
            board.push(game.move)

            # We are only interested in the centipawns value from the point of view of white.
            white_eval = evaluation.white()
            if white_eval.is_mate():
                continue

            # Write the results to stdout.
            res = f"{board.fen()};{white_eval.score()}"
            print(res)

            # Update stats.
            pos_count += 1
            bytes_count += len(res)

        # Print the advancement of the task.
        print(f"Position count: {pos_count // 1000}K ({bytes_count / 1000000:.2f}MB)", end="\r", file=sys.stderr)

if __name__ == "__main__":
    main()