from IPython.display import display
from ipycanvas import Canvas
from utils import *


class Interface:
    
    def __init__(self, agent) -> None:
        self.shape = (3, 3)
        self.board_size = 9
        self.row_size = 3
        
        self.board = np.array([0] * self.board_size).astype(np.float32)

        self.agent = agent

        self.human = "X" if self.agent.symbol == "O" else "O"

        self.title = f"Agent: {self.agent.symbol}"

        self.canvas = Canvas(width=600, height=300)
        self.canvas.font = '32px serif'

        self.canvas.on_mouse_down(self._click_handler)
        
        self.play()


    def play(self):
        if self.agent.symbol == "X":
            self._make_move(is_human=False)
        else:
            self._draw("X")

    
    def _click_handler(self, x, y):
        # human action
        outcome = self._make_move(is_human=True, x=x, y=y)            
        
        if outcome:
            return

        # agent action
        self._make_move(is_human=False)


    def _make_move(self, is_human=True, x=None, y=None):
        self.board *= -1
        if is_human:
            action = self._click_to_action(x, y)
            player = self.human
        else:
            action = self.agent.act(self.board)
            player = self.agent.symbol
            
        outcome, board, row = check_outcome(self.board, action, 
                                            self.shape, self.row_size)
        self.board = board
        
        self._draw(player, outcome, row)
        if outcome:
            self.canvas.on_mouse_down(self._click_handler, remove=True)
        return outcome


    def _click_to_action(self, x, y):
        row = int(y // 100)
        col = int(x // 100)
        board = np.arange(self.board_size).reshape(self.shape)
        action = board.item((row, col))
        return action


    def _draw(self, player, outcome="", row=None):
        symbolic_board = self._symbolic_board(player)
        
        self.canvas.clear()
        self.canvas.fill_text(self.title, 310, 25)
        self._draw_lines()
        self._draw_moves(symbolic_board)
        if outcome:
            self._draw_outcome(outcome, row)
        display(self.canvas)


    def _symbolic_board(self, player):
        X = 1 if player == "X" else -1
        symbolic_board = []
        for b in self.board:
            if b == 0:
                symbolic_board.append(" ")
            elif b == X:
                symbolic_board.append("X")
            else:
                symbolic_board.append("O")
        return np.reshape(symbolic_board, self.shape)


    def _draw_lines(self):
        self.canvas.stroke_style = 'black'
        self.canvas.stroke_line(100, 0, 100, 300)
        self.canvas.stroke_line(200, 0, 200, 300)
        self.canvas.stroke_line(0, 100, 300, 100)
        self.canvas.stroke_line(0, 200, 300, 200)

    def _draw_moves(self, board):
        RADIUS = 44
        LINE = 22

        for i, j in np.ndindex(board.shape):
            if not board[i, j]:
                continue

            x = j * 100 + 50
            y = i * 100 + 50

            if board[i, j] == "O":
                self.canvas.stroke_circle(x, y, RADIUS)
            elif board[i, j] == " ":
                pass
            else:
                self.canvas.stroke_line(x - LINE, y - LINE, x + LINE, y + LINE)
                self.canvas.stroke_line(x + LINE, y - LINE, x - LINE, y + LINE)

    def _draw_outcome(self, outcome, row):
        
        if outcome == "withdraw":
            self.canvas.fill_text("Draw!", 310, 150)
        elif outcome == "mistake":
            self.canvas.fill_text("Clicked on busy", 310, 150)
            self.canvas.fill_text("position.", 310, 200)
        else:
            self.canvas.stroke_style = 'red'
            a, b = reduce_row(row)
            x1 = int((a[1] + 1) * 100 - 50)
            y1 = int((a[0] + 1) * 100 - 50)
            x2 = int((b[1] + 1) * 100 - 50)
            y2 = int((b[0] + 1) * 100 - 50)
            self.canvas.stroke_line(x1, y1, x2, y2)
            self.canvas.fill_text("Game Over", 310, 150)
