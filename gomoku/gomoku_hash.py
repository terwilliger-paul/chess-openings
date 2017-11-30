import numpy as np

BOARD_SIZE = 19


def generate_win_hash(board_size):
    bitboard = np.array(list(range(board_size * board_size)), dtype=np.int16)
    board = bitboard.reshape((board_size, board_size))

    # Find horizontals
    horz = [np.array([np.array([a, b, c, d, e])
                      for a, b, c, d, e
                      in zip(r[0:], r[1:], r[2:], r[3:], r[4:])])
            for r in board]

    # Find verticals
    vert = [np.array([np.array([a, b, c, d, e])
                      for a, b, c, d, e
                      in zip(v[0:], v[1:], v[2:], v[3:], v[4:])])
            for v in board.T]

    # Find primary diagonals
    p_diag = [board.diagonal(i)
              for i in range(-board_size+1, board_size)
              if len(board.diagonal(i)) > 4]
    p_win = [np.array([np.array([a, b, c, d, e])
                      for a, b, c, d, e
                      in zip(r[0:], r[1:], r[2:], r[3:], r[4:])])
            for r in p_diag]

    # Find off diagonals
    b90 = np.rot90(board)
    o_diag = [b90.diagonal(i)
              for i in range(-board_size+1, board_size)
              if len(b90.diagonal(i)) > 4]
    o_win = [np.array([np.array([a, b, c, d, e])
                      for a, b, c, d, e in
                      zip(r[0:], r[1:], r[2:], r[3:], r[4:])])
            for r in o_diag]

    win_hash = np.vstack([
                          np.vstack(horz),
                          np.vstack(vert),
                          np.vstack(p_win),
                          np.vstack(o_win),
                          ])

    return win_hash

def generate_win_dict(board_size):
    win_hash = generate_win_hash(board_size)

    win_dict = {i: [] for i in range(board_size ** 2)}
    for row in win_hash:
        for i in row:
            win_dict[i].append(row)
    for key in win_dict:
        win_dict[key] = np.vstack(win_dict[key])

    return win_dict

'''
win_hash = generate_win_hash(BOARD_SIZE)
win_dict = generate_win_dict(BOARD_SIZE)
print(win_hash.size)
print(win_hash.nbytes)
print(np.sum(win_dict[i].nbytes for i in win_dict))
'''
