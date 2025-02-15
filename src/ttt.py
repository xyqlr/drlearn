import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
import logging
import argparse
import time
import os
from tqdm import tqdm

from utils import AverageMeter, dotdict
from arena import Arena
from mcts import MCTS
from agent import Agent

# Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self, n=3):
        self.n = n

    def get_init_state(self):
        # return initial board (numpy board)
        b = np.zeros((self.n*self.n), dtype=int)
        return b

    def get_shape(self):
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions
        return self.n*self.n + 1

    def get_next_state(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = np.copy(board)
        b[action] = player
        return (b, -player)

    def get_valid_actions(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.get_action_size()
        legalMoves =  list(zip(*np.where(board == 0)))
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for i in legalMoves:
            valids[i[0]]=1
        return np.array(valids)

    def is_win(self, board, player):
        # Check rows, columns, and diagonals
        if (np.any(np.all(board == player, axis=1)) or
            np.any(np.all(board == player, axis=0)) or
            np.all(np.diag(board) == player) or
            np.all(np.diag(np.fliplr(board)) == player)):
            return True
        return False
    
    def get_game_ended(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = np.copy(board).reshape(self.n, self.n)
        if self.is_win(b, player):
            return 1
        if self.is_win(b, -player):
            return -1
        if len(list(zip(*np.where(board == 0))))!=0:
            return 0
        # draw has a very little value 
        return 1e-4

    def get_player_agnostic_state(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def get_symmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        b2d = board.copy().reshape(self.n,self.n)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(b2d, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB.reshape(-1), list(newPi.ravel()) + [pi[-1]])]
        return l

    def state_to_string(self, board):
        # 8x8 numpy array (canonical board)
        return ','.join([str(board[x*self.n+y]) for x in range(self.n) for y in range(self.n)])

    @staticmethod
    def display(board):
        n = 3

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y*n+x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")

class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.get_valid_actions(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class NeuralNetModel(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_shape()
        self.action_size = game.get_action_size()
        self.args = args

        super().__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-2)*(self.board_y-2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)
        if args.cuda:
            super().cuda()

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-2)*(self.board_y-2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def fit(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.parameters())

        for epoch in range(self.args.epochs):
            logging.info('EPOCH ::: ' + str(epoch + 1))
            super().train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        super().eval()
        with torch.no_grad():
            pi, v = self(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            logging.debug("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            logging.debug("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])

args = dotdict({
    'numIters': 10,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'log_level': 'INFO',
    'test': False,
    'play': False,
    'num_games': 2,
})

nnargs = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

def main():
    loglevels = dict(DEBUG=logging.DEBUG, 
                     INFO=logging.INFO,
                     WARNING=logging.WARNING,
                     ERROR=logging.ERROR,
                     CRITICAL=logging.CRITICAL 
                     )
    logging.basicConfig(level=loglevels[args.log_level])

    game = TicTacToe()

    nnet = NeuralNetModel(game, nnargs)

    if args.test:
        logging.info('Playing against self')
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        mcts = MCTS(game, nnet, args)
        arena = Arena(lambda x: np.argmax(mcts.get_action_prob(x, temp=0)),
                        lambda x: np.argmax(mcts.get_action_prob(x, temp=0)), game)
        pwins, nwins, draws = arena.play_games(args.arenaCompare)

        logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    elif args.play:
        logging.info("Let's play!")
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        mcts = MCTS(game, nnet, args)
        cp = lambda x: np.argmax(mcts.get_action_prob(x, temp=0))
        hp = HumanTicTacToePlayer(game).play
        arena = Arena(cp, hp, game, display=TicTacToe.display)
        arena.play_games(args.num_games, verbose = True)

    else:
        if args.load_model:
            logging.info('Loading checkpoint "%s/%s"...', args.checkpoint, 'best.pth.tar')
            nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        else:
            logging.warning('Not loading a checkpoint!')

        c = Agent(game, nnet, args, nnargs)

        if args.load_model:
            logging.info("Loading 'trainExamples' from file...")
            c.load_train_examples(best=True)

        logging.info('Starting the learning process 🎉')
        c.learn()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Find starting indices of concatenated substrings in a string.")
    parser.add_argument("--load", action="store_true", help="load the last best checkpoint")
    parser.add_argument("--iters", type=int, help="number of iterations", default=5)
    parser.add_argument("--episodes", type=int, help="number of episodes/games for each iteration",default=1)
    parser.add_argument("--epochs", type=int, help="number of epochs for training",default=10)
    parser.add_argument("--loglevel", type=str, help="logging level",default='INFO')
    parser.add_argument("--test", action="store_true", help="test against self")
    parser.add_argument("--play", action="store_true", help="play against human")
    
    # Parse the arguments
    inargs = parser.parse_args()
    
    # Extract the input string and words
    args.numIters = inargs.iters
    args.numEps = inargs.episodes
    args.load_model = inargs.load
    args.log_level = inargs.loglevel
    args.test = inargs.test
    args.play = inargs.play
    nnargs.epochs = inargs.epochs

    main()
