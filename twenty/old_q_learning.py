import numpy as np
import cupy as cp
import chainer
from chainer import cuda, Function, gradient_check
from chainer import report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import time
import twenty_logic as tl

DTYPE = np.float32
GAMMA = .99

np.set_printoptions(precision=2)

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, new_qs):
        qs = self.predictor(x)
        loss = F.mean_squared_error(qs, new_qs)
        report({'loss': loss}, self)
        return loss

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_units)  # n_units -> n_units
            self.l5 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.tanh(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        h3 = F.tanh(self.l3(h2))
        h4 = F.tanh(self.l4(h3))
        y = self.l5(h4)
        return y

def cp_q_rot90(cp_q):
    cp_out = cp.zeros(4, dtype=cp_q.dtype)
    cp_out[0] = cp_q[2]
    cp_out[1] = cp_q[0]
    cp_out[2] = cp_q[3]
    cp_out[3] = cp_q[1]
    return cp_out

def cp_q_t(cp_q):
    cp_out = cp.zeros(4, dtype=cp_q.dtype)
    cp_out[0] = cp_q[2]
    cp_out[1] = cp_q[3]
    cp_out[2] = cp_q[0]
    cp_out[3] = cp_q[1]
    return cp_out


def d4_board(cp_board, cp_q):
    '''left becomes down'''
    b = cp_board
    q = cp_q
    b90 = cp.rot90(cp_board)
    q90 = cp_q_rot90(cp_q)
    b180 = cp.rot90(b90)
    q180 = cp_q_rot90(q90)
    b270 = cp.rot90(b180)
    q270 = cp_q_rot90(q180)

    output = [(b, q),
              (b90, q90),
              (b180, q180),
              (b270, q270),
              (cp.transpose(b), cp_q_t(q)),
              (cp.transpose(b90), cp_q_t(q90)),
              (cp.transpose(b180), cp_q_t(q180)),
              (cp.transpose(b270), cp_q_t(q270)),
              ]

    output = [(o[0].reshape((25)), o[1]) for o in output]

    return output

def daytime(model, eps=.01, replay_len=1000, logging=False,
            additional_memory=[], target=1024):

    scores = []
    lengths = []

    # Set up one daytime
    start = time.time()
    replay_memory = []
    #while len(replay_memory) < replay_len:
    for _ in range(1):
        # Play a whole game
        Game = tl.Game()
        length = 0
        while Game.game_end == False:
            length += 1
            cp_board = cp.array(Game.board.reshape((1, 25)), copy=True)
            qs = model.predictor(cp_board).data[0]
            move = np.int(cp.argmax(qs).get())

            # In numpy
            # Random action
            #move = np.random.choice([move, np.random.randint(0, 4)], p=[1-eps, eps])

            q = qs[move]
            new_board = tl.fast_move(move, Game.board)

            # Illegal move
            if tl.fast_all(Game.board, new_board):
                move = np.random.randint(0, 4)
                q = qs[move]
                new_board = tl.fast_move(move, Game.board)

            score = tl.fast_sum(new_board)
            r = score
            '''
            r = 0
            if score > target:
                r = 1
            '''
            boards = tl.gen_all_boards(new_board)

            # Illegal move with no open squares
            maxq = 0
            if len(boards) == 0:
                maxq = q
            else:
                maxq = np.min([np.max(model.predictor(cp.array(board.reshape((1, 25)))).data.get())
                                  for board in boards])
            new_q = r + (GAMMA*maxq)
            new_qs = qs.copy()
            new_qs[move] = new_q
            new_board_reshape = new_board.reshape((25))
            '''
            for i in range(25):
                if new_board_reshape[i] == 0:
                    new_board_reshape[i] = 1
            new_board_reshape = np.log2(new_board_reshape)
            '''

            # Add state to replay memory
            replay_memory.append((cp.array(new_board), new_qs))

            # Do move
            Game.game_move(move)
        scores.append(Game.score)
        lengths.append(length)
        print(Game.score, length)
    print(time.time() - start)

    memory = []
    for cp_board, cp_q in replay_memory:
        memory += d4_board(cp_board, cp_q)

    '''
    if logging:
        print(np.mean(scores), np.std(scores), np.max(scores))
    '''

    return additional_memory + memory, scores, lengths

def nighttime(replay_memory, optimizer):
    # Nighttime is when it learns
    train_iter = iterators.SerialIterator(replay_memory, batch_size=128, shuffle=True)
    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

model = Classifier(MLP(2048, 4))
optimizer = optimizers.SMORMS3()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(.0001))

replay_memory = [(cp.random.randn(25, dtype=cp.float32),
                  cp.random.randn(4, dtype=cp.float32)) for _ in range(1000)]
nighttime(replay_memory, optimizer)
all_scores = [1024]
scores = []
lengths = []
for _ in range(10000):
    replay_memory, score, length = daytime(model, logging=True, target=np.mean(all_scores),
                                           additional_memory=replay_memory[-20000:])
    all_scores = all_scores + score
    scores.append([np.mean(score), np.std(score), np.max(score)])
    lengths.append([np.mean(length), np.std(length), np.max(length)])
    print(np.hstack([np.array(scores), np.array(lengths)]))
    nighttime(replay_memory, optimizer)
