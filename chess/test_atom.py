import numpy as np
import chess
from chess import pgn

def read_pgn(filename):
    with open(filename) as f:
        game = pgn.read_game(f)
    return game

filename = 'white_repertoire.pgn'
filename = "C:/Users/585191/Google Drive/Python/chess/" + filename
print(read_pgn(filename))