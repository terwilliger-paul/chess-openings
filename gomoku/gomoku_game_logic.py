import gomoku_hash
import numpy as np

class Game:

    def __init__(self, board_size):
        self.board_size = board_size
        self.bitboard = np.zeros([board_size ** 2], dtype=np.int16)
        self.board = self.bitboard.reshape((board_size, board_size))
        self.win_dict = gomoku_hash.generate_win_dict(board_size)
        self.to_move = 1
        self.game_end = False
        self.victory = 0

    def move(self, coord):

        if self.game_end == True:
            return

        if self.bitboard[coord] != 0:
            self.game_end = True
            self.victory = self.to_move * -1
        else:
            # Make the move
            self.bitboard[coord] = self.to_move

            # Check for game end
            # victory first
            for row in self.win_dict[coord]:
                is_victory = np.sum(self.bitboard[i] for i in row)
                if is_victory == 5:
                    self.game_end = True
                    self.victory = 1
                elif is_victory == -5:
                    self.game_end = True
                    self.victory = -1
            # board full second
            if 0 not in self.bitboard:
                self.game_end = True
                self.victory = 0

            self.to_move *= -1
