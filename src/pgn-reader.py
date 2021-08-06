from io import StringIO
from sys import stdin, stderr

import chess, chess.pgn

# A leaner extractor that only reads moves and comments, and skips everything if it encounters
# an unevaluated game.
class FastGameBuilder(chess.pgn.GameBuilder):
    def begin_game(self):
        self.game = self.Game()
        self.variation_stack = [self.game]
        self.first = True

    def begin_headers(self):
        pass

    def visit_header(self, tagname, tagvalue):
        pass

    def visit_nag(self, nag):
        pass

    def begin_variation(self):
        pass

    def end_variation(self):
        pass

    def visit_result(self, result):
        pass

    def visit_comment(self, comment):
        self.variation_stack[0].comment = comment
        if self.first and self.variation_stack[0].eval() is None:
            raise Skip()
        self.first = False

    def visit_move(self, board: chess.Board, move: chess.Move):
        self.variation_stack[0] = self.variation_stack[0].add_variation(move=move)

    def handle_error(self, error):
        raise error

    def result(self):
        return self.game

# Inherits BaseException because Exceptions are swallowed by python-chess.
class Skip(BaseException):
    pass

def get_chunk():
    lines = []
    after_headers = False

    for line in stdin:
        if line.isspace():
            if after_headers:
                break
            after_headers = True
        lines.append(line)

    return StringIO("".join(lines))

# Reads PGN games from stdin and converts them into evaluated FEN positions.
def main():
    # The chess board.
    board = chess.Board()

    # Keeps track of the number of position extracted.
    pos_count = 0
    bytes_count = 0

    # For each game provided through stdin.
    while True:
        try:
            chunk = get_chunk()
            game = chess.pgn.read_game(chunk, Visitor=FastGameBuilder)
            chunk.close()
        except Skip:
            continue

        if game is None:
            break

        # Reset the board to it's starting position.
        board.reset()

        # For each position in the game.
        while game := game.next():
            # Get the evaluation on this node. Not all game are evaluated sadly.
            if not (evaluation := game.eval()):
                continue

            # Push that move to the board.
            board.push(game.move)

            # We are only interested in the centipawns value from the point of view of white.
            white_eval = evaluation.white()
            if white_eval.is_mate():
                continue

            res = f"{board.fen()};{white_eval.score()}"
            print(res)
            
            # Update stats.
            pos_count += 1
            bytes_count += len(res)+1        

        # Print the advancement of the task.
        print(f"Position count: {pos_count // 1000}K ({bytes_count / 1000000:.2f}MB)", end="\r", file=stderr)

if __name__ == "__main__":
    main()