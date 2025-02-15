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

# Custom Blackjack Environment
class Blackjack:
    def __init__(self):
        self.n = 13
        self.reset()

    def reset(self):
        # Reset the deck
        self.suite = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.deck = self.suite * 4  # Deck of cards
        random.shuffle(self.deck)
        
        # Reset player and dealer hands
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.index_map = {k:i for i,k in enumerate(self.suite)}
        # Card values: Ace=1, 2=2, ..., 10=10, Jack=10, Queen=10, King=10
        self.card_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

    def get_init_state(self):
        # State: 1D vector of size 13 (count of each card in player's hand)
        state = np.zeros(self.n, dtype=int)
        for card in self.player_hand:
            state[self.index_map[card]] += 1  # Cards are 1-indexed (1 to 13)
        return state

    def get_shape(self):
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions: 0 for hit and 1 for stand
        return 2

    def _get_value(self, state):
        """
        Compute all possible values of a Blackjack hand using a NumPy array of card counts.
        
        Args:
            state (np.array): Array of card counts, where:
                                    - card_counts[0] = number of Aces
                                    - card_counts[1] = number of 2s
                                    - ...
                                    - card_counts[12] = number of Kings
        
        Returns:
            set: A set of possible hand values.
        """
        
        # Base value: treat all Aces as 1
        base_value = np.sum(state * self.card_values)
        
        # Number of Aces
        num_aces = state[0]
        
        if num_aces ==0 or base_value > 21:
            return [base_value]
        # Possible values: add 10 for each Ace that can be treated as 11
        possible_values = [base_value + 10 * i for i in range(num_aces + 1) if base_value + 10 * i <= 21]
        
        return possible_values
    
    def get_next_state(self, state, player, action):
        s = np.copy(state)
        if action == 0:  # Hit
            card = self.deck.pop()
            self.player_hand.append(card)
            s[self.index_map[card]] += 1
            return (s, player)
        else:  # Stand
            dealer_state = np.zeros(self.n, dtype=int)
            for card in self.dealer_hand:
                dealer_state[self.index_map[card]] += 1  # Cards are 1-indexed (1 to 13)
            dealer_values = self._get_value(dealer_state)
            while max(dealer_values) < 17:  # Dealer's policy
                card = self.deck.pop()
                self.dealer_hand.append(card)
                dealer_state[self.index_map[card]] += 1
                dealer_values = self._get_value(dealer_state)
            dealer_sum = max(dealer_values)
            player_values = self._get_value(state)
            player_sum = max(player_values)
            if dealer_sum > 21 or player_sum > dealer_sum:
                return state, 1
            elif player_sum == dealer_sum:
                return state, 0
            else:
                return state, -1

    def get_valid_actions(self, state, player):
        # return a fixed size binary vector
        valids = [1]*self.get_action_size()
        possible_values = self._get_value(state)
        if min(possible_values) > 21:  # Bust
            valids[0]=0
        return np.array(valids)

    def get_game_ended(self, state, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        possible_values = self._get_value(state)
        if min(possible_values) > 21:  # Bust
            return -1
        else:
            return 0

    def get_player_agnostic_state(self, state, player):
        # return state if player==1, else return -state if player==-1
        return player*state

    def get_symmetries(self, state, pi):
        return [state, pi]

    def state_to_string(self, state):
        return ','.join([self.suite[x] for y in range(state[x])] for x in range(self.n) if state[x]>0)

    @staticmethod
    def display(state):
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
                piece = state[y*n+x]    # get the piece to print
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

class HumanBlackJackPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        # display(state)
        valid = self.game.get_valid_actions(state, 1)
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
        self.state_size = game.get_shape()[0]
        self.action_size = game.get_action_size()
        self.args = args
        super().__init__()
        self.fc1 = nn.Linear(self.state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def fit(self, examples):
        """
        examples: list of examples, each example is of form (state, pi, v)
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

    def predict(self, state):
        """
        state: np array with state
        """
        # timing
        start = time.time()

        # preparing input
        state = torch.FloatTensor(state.astype(np.float64))
        if self.args.cuda: state = state.contiguous().cuda()
        state = state.view(1, self.state_size)
        super().eval()
        with torch.no_grad():
            pi, v = self(state)

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
