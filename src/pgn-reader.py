from io import StringIO
from multiprocessing import cpu_count, Process, Queue
from queue import Empty, Full
from sys import stdin

from chess import Board
from chess.pgn import GameBuilder, read_game

"""
Reads PGN data from stdin, extract the positions that are evaluated and 
outputs them to stdout, one by line, in the following format 'FEN;EVAL'.

Where FEN is the FEN representation of the board and EVAL is a signed 
integer giving the evaluation of the board from white's point of viewn,
in centipawns.

Ignores positions that are evaluated with a mate score.

Skips games that are not evaluated.
"""

# The numbers of moves to skip from the start of the game.
SKIP_NFIRST = 4

# The maximum capacity in queues.
MAX_QUEUE = 8

class FastGameBuilder(GameBuilder):
    """
    A leaner PGN visitor that only reads moves and comments, and skips everything 
    if it encounters an unevaluated game.
    """

    def begin_game(self):
        self.game = self.Game()
        self.variation_stack = [self.game]
        self.first = True

    def visit_comment(self, comment):
        self.variation_stack[0].comment = comment
        if self.first and self.variation_stack[0].eval() is None:
            raise Skip()
        self.first = False

    def visit_move(self, board, move):
        self.variation_stack[0] = self.variation_stack[0].add_variation(move=move)

    def handle_error(self, error):
        raise Skip()

    def begin_headers(self): pass
    def visit_header(self, tagname, tagvalue): pass
    def visit_nag(self, nag): pass
    def begin_variation(self): pass
    def end_variation(self): pass
    def visit_result(self, result): pass

class Skip(BaseException):
    """
    Used to signal that we want to skip this game.
    Inherits BaseException because normal Exception classes are swallowed by python-chess.
    """

    pass

def print_results(res_queue):
    """
    Prints resulst to stdout until the pipe is broken or nothing 
    came for one second straight.
    """

    try:
        # Outputs the results.
        while True:
            for res in res_queue.get(timeout=1):
                print(res)
    except (BrokenPipeError, Empty):
        return

def worker_main(in_queue, res_queue):
    """
    Parses games sent from in_queue and outputs evaluated FEN positions
    to res_queue.

    Skips the first SKIP_NFIRST moves.
    """

    board = Board()

    while True:
        try:
            chunk = in_queue.get(timeout=1)
        except Empty:
            return
        
        try:
            game = read_game(chunk, Visitor=FastGameBuilder)
        except Skip:
            continue
        finally:
            chunk.close()

        # End reached.
        if game is None:
            break

        # Reset the board to it's starting position.
        board.reset()
        batch = []
        n = 0

        # For each position in the game.
        while game := game.next():
            n += 1

            # Do the move
            board.push(game.move)

            if n < SKIP_NFIRST:
                continue

            # Position is not evaluated
            if not (evaluation := game.eval()):
                continue
            # Get white's point of view.
            white_eval = evaluation.white()
            # Evaluation is a mate score.
            if white_eval.is_mate():
                continue

            # Format it like: FEN;EVAL".
            batch.append(f"{board.fen().split(' ')[0]};{white_eval.score()}")

        # Try to push the next results to the queue but quits if no
        # one is there to consume.
        try:
            res_queue.put(batch, timeout=1)
        except Full:
            return

def main():
    """
    Creates the different processes of the program, then start reading from
    stdin and send the games to the parser processes, in pure text.
    """

    in_queue = Queue(maxsize=MAX_QUEUE)
    res_queue = Queue(maxsize=MAX_QUEUE)

    Process(
        target=print_results,
        kwargs={
            "res_queue": res_queue,
        },
    ).start()

    for _ in range(max(cpu_count() - 2, 1)):
        Process(
            target=worker_main,
            kwargs={
                "in_queue": in_queue,
                "res_queue": res_queue,
            },
        ).start()

    # Reads lines from stdin until there is nothing left, 
    # or the pipe is broken, or no one is consuming games
    # anymore.
    try:
        lines = []
        after_headers = False

        for line in stdin:
            if line.isspace():
                if after_headers:
                    in_queue.put(StringIO("".join(lines)), timeout=1)
                    lines = []
                    after_headers = False
                else:
                    after_headers = True
            lines.append(line)
    except (BrokenPipeError, Full):
        return

if __name__ == "__main__":
    main()