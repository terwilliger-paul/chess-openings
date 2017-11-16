import chess
from chess import pgn
from chess import svg
import re
import numpy as np
from IPython.display import SVG, display
from sys import version_info
import datetime
import chess_module as cm
import pandas as pd

LIVES_RE = '\[%half_life (.+?)]'
DATES_RE = '\[%datetime (.+?)]'

LIVES_TEXT = " [%half_life {}]"
DATES_TEXT = " [%datetime {}]"

class ChessError(Exception):
    pass

#creates boolean value for test that Python major version > 2
py3 = version_info[0] > 2

def show_svg(svg):
    return display(SVG(svg))

def read_pgn(filename):
    with open(filename) as f:
        game = pgn.read_game(f)
    return game

def create_chess_tree(game):
    """Create a chess tree"""

    tree = [game]
    moves = [[]]
    flag = True

    # Iterate through all variations in the tree
    while flag:
        flag = False
        for b_i, b in enumerate(tree):
            if len(b.variations) > 0:
                flag = True
                sub_pgn = tree.pop(b_i)
                sub_tree = sub_pgn.variations
                sans = [a.san() for a in sub_tree]
                sub_move = moves.pop(b_i)
                sub_moves = [sub_move[:] + [s_i] for s_i, san
                             in enumerate(sans)]

                for t_i in range(len(sub_tree)):
                    tree.append(sub_tree[t_i])
                    moves.append(sub_moves[t_i])

    return moves, tree

def load_weights(moves, tree):

    # Read weights
    weights = np.array([1. for _ in moves])
    for e_i, end in enumerate(tree):
        text = end.comment
        m = re.search('\[%weight (.+?)]', text)
        if m:
            found = m.group(1)
            weights[e_i] = float(found)
            end.comment = re.sub('\[%weight (.+?)]', '', text)

    # Decay weights
    if 'mem_date' in game.headers:
        delta = (datetime.date.today() - datetime.datetime.strptime(
                        game.headers['mem_date'], "%m/%d/%Y").date()).days
        for _ in range(delta):
#            weights *= .8
            weights -= .25

    # Correct for weights below 1
    weights[weights < 1.] = 1.

    return weights

def load_scores(moves, tree):

    # Read scores
    old_date = datetime.datetime.today() - datetime.timedelta(365)
    lives = np.array([cm.HALF_LIFE for _ in moves])
    dates = np.array([old_date for _ in moves])

    for e_i, end in enumerate(tree):
        text = end.comment

        # Find streaks
        m = re.search(DATES_RE, text)
        if m:
            found = m.group(1)
            dates[e_i] = pd.to_datetime(found)#.to_pydateime()
            end.comment = re.sub(DATES_RE, '', text)

    for e_i, end in enumerate(tree):
        text = end.comment

        # Find streaks
        m = re.search(LIVES_RE, text)
        if m:
            found = m.group(1)
            lives[e_i] = float(found)
            end.comment = re.sub(LIVES_RE, '', text)

    '''
    weights = np.array([1. for _ in moves])
    streaks = np.array([0. for _ in moves])

    for e_i, end in enumerate(tree):
        text = end.comment

        # Find streaks
        m = re.search('\[%streak (.+?)]', text)
        if m:
            found = m.group(1)
            streaks[e_i] = float(found)
            end.comment = re.sub('\[%streak (.+?)]', '', text)

    for e_i, end in enumerate(tree):
        text = end.comment
        # Find weights
        m = re.search('\[%weight (.+?)]', text)
        if m:
            found = m.group(1)
            weights[e_i] = float(found)
            end.comment = re.sub('\[%weight (.+?)]', '', text)

    # Decay weights
    if 'mem_date' in game.headers:
        delta = (datetime.date.today() - datetime.datetime.strptime(
                        game.headers['mem_date'], "%m/%d/%Y").date()).days
        for _ in range(delta):
#            weights *= .8
            weights -= 1

    # Correct for weights below 1
    weights[weights < 1.] = 1.

    return weights, streaks
    '''

    return dates, lives

# Uses `weights` and `streaks`
'''
def test_variation(game, moves, weights, streaks, white):
    correct = True
    my_turn = white
    temp_weights = weights.copy()
    choice = np.random.choice(range(len(moves)),
                              p=((1./weights) / np.sum(1./weights)))
    test = moves[choice]
    current_pos = game
    prompt = "Guess the move: "
    i = 0
    while i < len(test):
        c_i = test[i]
        positions = [v.san() for v in current_pos.variations]
        this_turn = True
        if my_turn:
            show_svg(svg.board(current_pos.board(), flipped=not white))
            if py3:
                guess = input(prompt)
            else:
                guess = raw_input(prompt)
            if guess in ['exit', 'quit', 'break']:
                raise ChessError
            guess = guess.lower().strip()
            positions = [p.lower().strip() for p in positions]
            if guess == 'boost':
                streaks += 2
                weights += 2
                return weights, streaks
            if guess == "":
                this_turn = False
                print("pass")
            elif guess in ["true", "correct"]:
                correct = True
                this_turn = False
                print(">> correct == True")
            elif guess == "hint":
                correct = False
                this_turn = False
                print(positions)
            elif not np.any([((guess in pos) and (len(guess) > 1))
                             for pos in positions]):
                correct = False
                this_turn = False
                print("Incorrect, try again")
        if this_turn:
            my_turn = not my_turn
#            print(positions)
            current_pos = current_pos.variations[c_i]
            i += 1

    show_svg(svg.board(current_pos.board(), flipped=not white))
    old_weight = weights[choice]
    old_streak = streaks[choice]
    if correct:
        print("Correct!")
        weights[choice] *= 2.
        weights[choice] += streaks[choice]
        streaks[choice] += 1
    else:
        weights[choice] /= 2.
        streaks[choice] = 0.

    # Correct for weights below 1
    weights[weights < 1.] = 1.

#    # Recursion
#    recursion_prompt = "test again? "
#    if py3:
#        guess = input(recursion_prompt)
#    else:
#        guess = raw_input(recursion_prompt)
#
#    if guess == 'y':
#        weights = test_variation(game, moves, weights, white=white)

    new_weight = weights[choice]
    new_streak = streaks[choice]
    print('weight changed from {} to {}'.format(old_weight, new_weight))
    print('streak changed from {} to {}'.format(old_streak, new_streak))
    return weights, streaks
'''

def test_variation(game, moves, dates, lives, white):
    correct = True
    my_turn = white

    probs = cm.dates_to_prob(dates, lives)

    if np.sum(probs) < 1e-7:
        choice = np.random.choice(range(len(moves)))
        print("sum of probs:", np.sum(probs))
    else:
        choice = np.random.choice(range(len(moves)),
                                  p=((1./probs) / np.sum(1./probs)))

    test = moves[choice]
    current_pos = game
    prompt = "Guess the move: "
    i = 0
    while i < len(test):
        c_i = test[i]
        positions = [v.san() for v in current_pos.variations]
        this_turn = True
        if my_turn:
            show_svg(svg.board(current_pos.board(), flipped=not white))
            if py3:
                guess = input(prompt)
            else:
                guess = raw_input(prompt)
            if guess in ['exit', 'quit', 'break']:
                raise ChessError
            guess = guess.lower().strip()
            positions = [p.lower().strip() for p in positions]
            if guess == "":
                this_turn = False
                print("pass")
            elif guess in ["true", "correct"]:
                correct = True
                this_turn = False
                print(">> correct == True")
            elif guess == "hint":
                correct = False
                this_turn = False
                print(positions)
            elif not np.any([((guess in pos) and (len(guess) > 1))
                             for pos in positions]):
                correct = False
                this_turn = False
                print("Incorrect, try again")
        if this_turn:
            my_turn = not my_turn
#            print(positions)
            current_pos = current_pos.variations[c_i]
            i += 1

    show_svg(svg.board(current_pos.board(), flipped=not white))
    old_date = dates[choice]
    old_life = lives[choice]
    if correct:
        print("Correct!")
        print(cm.new_half_life(old_date, old_life))
        dates[choice] = datetime.datetime.today()
    else:
        print("Failure.")

#    # Recursion
#    recursion_prompt = "test again? "
#    if py3:
#        guess = input(recursion_prompt)
#    else:
#        guess = raw_input(recursion_prompt)
#
#    if guess == 'y':
#        weights = test_variation(game, moves, weights, white=white)

    new_date = dates[choice]
    new_life = lives[choice]
    print('date changed from {} to {}'.format(old_date, new_date))
    print('life changed from {} to {}'.format(old_life, new_life))
    return dates, lives

# Uses `weights` and `streaks`
'''
def write_pgn(filename, game, tree, weights, streaks):

    for e_i, end in enumerate(tree):
        text = end.comment
        end.comment = text + \
                      " [%weight {}]".format(weights[e_i]) + \
                      " [%streak {}]".format(streaks[e_i])
    game.headers['mem_date'] = datetime.date.today().strftime("%m/%d/%Y")
    with open(filename, 'w') as f:
        f.write(str(game))
'''

def write_pgn(filename, game, tree, dates, lives):

    for e_i, end in enumerate(tree):
        text = end.comment
        end.comment = text + \
                      LIVES_TEXT.format(lives[e_i]) + \
                      DATES_TEXT.format(dates[e_i])
    game.headers['mem_date'] = datetime.date.today().strftime("%m/%d/%Y")
    with open(filename, 'w') as f:
        f.write(str(game))


# Load game
white = np.random.choice([True, False])
if white == True:
    filename = 'white_repertoire.pgn'
else:
    filename = 'black_repertoire.pgn'
#filename = "C:/Users/585191/Google Drive/Python/chess/" + filename
game = read_pgn(filename)
print(game)

# Calculate all variations
moves, tree = create_chess_tree(game)

# Load weights
dates, lives = load_scores(moves, tree)

print(dates)
print(lives)

'''
print('weights', np.sum(weights), 'max', np.max(weights), 'lines', len(weights))
print(weights)
print(streaks)
'''

# Pick a variation
dates, lives = test_variation(game, moves, dates, lives, white)

# Put weights back into pgn
write_pgn(filename, game, tree, dates, lives)
print("Done")
