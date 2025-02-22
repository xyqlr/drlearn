import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import argparse
import time

from rlearn.game import Game
from rlearn.nnet import NeuralNetModel
from rlearn.args import args, nnargs, parse_args, main

# Tic-Tac-Toe Environment
class TicTacToe(Game):
    def __init__(self, n=3):
        super().__init__()
        self.n = n

    def get_init_state(self):
        '''
        state: {state, state, current_player, reward)
        reward:  1    :  if current_player wins
                -1    :  if the other player wins
                0     :  game not finished
                1e-4  :  game is tied
        '''
        b = np.zeros((self.n*self.n), dtype=int)
        return b, b, 1, 0

    def to_neural_state(self, state):
        '''
        convert to numpy, which can be fed to the neural network
        no need to convert here, as it is always in numpy
        '''
        return state

    def get_shape(self):
        '''
        dimensions of the state: (3,3)
        '''
        return (self.n, self.n)

    def get_action_size(self):
        '''
        the number of actions: each actioin is represented by the index to the board
        the last action is pass
        '''
        return self.n*self.n + 1

    def get_next_state(self, state, player, action):
        '''
        this is the critical API which controls the state transition of the game.
        if player takes action on state, return the next state
        action must be a valid move
        '''
        if action == self.n*self.n:
            return state[0], state[1], -state[2], state[3]
        next = np.copy(state[0])
        next[action] = player
        b = np.copy(next).reshape(self.n, self.n)
        if self._is_win(b, player):
            reward = 1
        elif self._is_win(b, -player):
            reward = -1
        elif len(list(zip(*np.where(next == 0))))!=0:
            reward = 0
        else:
            # draw has a very little value 
            reward = 1e-4
        return next, next, -player, reward 

    def get_valid_actions(self, state, player):
        '''
        given the current state, return the valid vector of actions
        '''
        state0 = state[0]
        valids = [0]*self.get_action_size()
        legal_moves =  list(zip(*np.where(state0 == 0)))
        if len(legal_moves)==0:
            valids[-1]=1
            return np.array(valids)
        for i in legal_moves:
            valids[i[0]]=1
        return np.array(valids)

    def get_player_agnostic_state(self, state, player):
        '''
        a state that is agonistic to the player, which is fed to the neural network for both players
        return state if player==1, else return -state if player==-1
        '''
        state0 = state[0]*player
        return state0, state0, state[2], state[3]

    def get_symmetries(self, state, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        state0 = state[0]
        b2d = state0.copy().reshape(self.n,self.n)
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

    def get_game_ended(self, state, player):
        '''
        this returns the ending status of the game:
            1   : if the player wins
            -1  : if the player loses
            0   : a tie
            1e-4: game not ended
        '''
        b = np.copy(state[0]).reshape(self.n, self.n)
        if self._is_win(b, player):
            reward = 1
        elif self._is_win(b, -player):
            reward = -1
        elif len(list(zip(*np.where(b == 0))))!=0:
            reward = 0
        else:
            # draw has a very little value 
            reward = 1e-4
        return reward 

    def state_to_string(self, state):
        '''
        string representation of the state
        '''
        s = state[0]
        return ','.join([str(s[x*self.n+y]) for x in range(self.n) for y in range(self.n)])

    @staticmethod
    def display(state):
        n = 3
        state0 = state[0]
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
                piece = state0[y*n+x]    # get the piece to print
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

    def _is_win(self, state, player):
        # Check rows, columns, and diagonals
        if (np.any(np.all(state == player, axis=1)) or
            np.any(np.all(state == player, axis=0)) or
            np.all(np.diag(state) == player) or
            np.all(np.diag(np.fliplr(state)) == player)):
            return True
        return False
    

class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        # display(state)
        valid = self.game.get_valid_actions(state, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True: 
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class TicTacToeModel(NeuralNetModel):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_shape()
        self.action_size = game.get_action_size()
        super().__init__(game, args)

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

    def predict(self, state):
        """
        state: np array with state
        """
        # timing
        start = time.time()

        # preparing input
        state = torch.FloatTensor(state.astype(np.float64))
        if self.args.cuda: state = state.contiguous().cuda()
        state = state.view(1, self.board_x, self.board_y)
        super().eval()
        with torch.no_grad():
            pi, v = self(state)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

if __name__ == "__main__":
    nnargs.channels = 64     #set the default
    args.numEps=10
    parse_args()
    game = TicTacToe()
    nnet = TicTacToeModel(game, nnargs)
    human_player = HumanTicTacToePlayer(game)
    main(game, nnet, human_player)