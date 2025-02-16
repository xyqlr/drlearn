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
import math
import sys
from tqdm import tqdm
from random import shuffle
from collections import deque
from pickle import Pickler, Unpickler

from utils import AverageMeter, dotdict

EPS = 1e-8

# Custom Blackjack Environment
class BlackJack:
    def __init__(self):
        self.n = 13
        self.reset()

    def reset(self):
        # Reset the deck
        self.suite = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.deck = self.suite * 4  # Deck of cards
        random.shuffle(self.deck)
        
        # Reset player and dealer hands
        self.current_player = 1 #player, -1 for dealer
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.index_map = {k:i for i,k in enumerate(self.suite)}
        # Card values: Ace=1, 2=2, ..., 10=10, Jack=10, Queen=10, King=10
        self.card_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

    def get_init_state(self):
        self.reset()
        return self.player_hand, self.dealer_hand, self.current_player, 0
    
    def to_state_np(self, state):
        #input state: (player_hand, dealer_hand, current_player, reward)
        #output state: 1D vector of size 14 (count of each card in player's hand)
        # for player the last one being the first card of the dealer, and for dealer the value of the player
        current_player = state[2]
        player, dealer = 0, 1
        if current_player == -1:
            player, dealer = 1, 0
        player_state = np.zeros(self.n+1, dtype=int)
        for card in state[player]:
            player_state[self.index_map[card]] += 1  # Cards are 1-indexed (1 to 13)
        player_state[self.n] = self.index_map[state[dealer][0]] #the first card of the dealer
        dealer_state = np.zeros(self.n+1, dtype=int)
        for card in state[dealer]:
            dealer_state[self.index_map[card]] += 1  # Cards are 1-indexed (1 to 13)
        dealer_state[self.n] = max(self._get_value(player_state))
        return (player_state, dealer_state, state[2], state[3]) if current_player == 1 else (dealer_state, player_state, state[2], state[3])

    def get_shape(self):
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions: 0 for hit and 1 for stand
        return 2

    def _deal_next_card(self, state):
        """
        Perform weighted sampling from a dictionary.

        Args:
            state: A tuple with player's and dealer's cards.

        Returns:
            str: a card.
        """

        # get the remainng cards
        cards = {k:4 for k in self.suite}
        for hand in range(2):
            for card in state[hand]:
                cards[card] -= 1

        # Extract keys and weights
        keys = list(cards.keys())
        weights = list(cards.values())

        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Perform weighted sampling
        card = random.choices(keys, weights=probabilities, k=1)
        return card[0]
        
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
        base_value = np.sum(state[:-1] * self.card_values) #exclude the last one
        
        # Number of Aces
        num_aces = state[0]
        
        if num_aces ==0 or base_value > 21:
            return [base_value]
        # Possible values: add 10 for each Ace that can be treated as 11
        possible_values = [base_value + 10 * i for i in range(num_aces + 1) if base_value + 10 * i <= 21]
        
        return possible_values
    
    def get_next_state(self, state, player, action):
        state0 = copy.copy(state[0])
        state1 = copy.copy(state[1])
        if action == 0:  # Hit
            card = self._deal_next_card(state)
            state0.append(card)
            state_np, _, _, _ = self.to_state_np((state0, state1, state[2], state[3]))
            if min(self._get_value(state_np)) > 21:
                #reward = -1 if player == 1 else 1
                return state0, state1, state[2], -player
            else:
                return state0, state1, state[2], state[3]
        else:  # Stand
            if player == 1:
                return state1, state0, -1, 0    #dealer's turn
            dealer_state, player_state, _, _ = self.to_state_np(state)
            dealer_sum = max(self._get_value(dealer_state))
            player_sum = max(self._get_value(player_state))
            if player_sum > dealer_sum:
                return state0, state1, state[2], -player
            elif player_sum == dealer_sum:
                return state0, state1, state[2], 1e-4 #small value for tie
            else:
                return state0, state1, state[2], player

    def get_valid_actions(self, state, player):
        valids = [1]*self.get_action_size()
        current_player = state[2]
        if current_player == -1: #dealer
            dealer_state, player_state, _, _ = self.to_state_np(state)
            dealer_sum = max(self._get_value(dealer_state))
            if dealer_sum < 17:
                valids[1] = 0            
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
        return state

    def get_symmetries(self, state, pi):
        return [state, pi]

    def state_to_string(self, state):
        current_player = state[2]
        if current_player == 1:
            dealer_str = str(self.index_map[state[1][0]])
            player_state, _, _, _ = self.to_state_np(state)
            return ''.join([str(v) for k,v in enumerate(player_state[:-1])])+":"+dealer_str
        else:
            dealer_state, player_state, _, _ = self.to_state_np(state)
            return ''.join([str(v) for k,v in enumerate(player_state[:-1])])+":"+''.join([str(v) for k,v in enumerate(dealer_state[:-1])])

    @staticmethod
    def display(state):
        player_state, dealer_state, current_player, _ = state
        if current_player == -1:
            dealer_state, player_state, _, _ = state
            dealer_str = ','.join(x for x in dealer_state)
        else:
            dealer_str = dealer_state[0]
        print(f"dealer: {dealer_str}")
        player_str = ','.join(x for x in player_state)
        print(f"player: {player_str}\n")

class HumanBlackJackPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        # display(state)
        valid = self.game.get_valid_actions(state, 1)
        str = f"Enter 0 for hit\n" if valid[0] else f""
        str += f"Enter 1 for stand" if valid[1] else f""
        print(str)
        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()
            a = int(a)
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class NeuralNetModel(nn.Module):
    def __init__(self, game, args):
        # game params
        self.state_size = game.get_shape()[0]+1
        self.action_size = game.get_action_size()
        self.args = args
        super().__init__()
        self.fc1 = nn.Linear(self.state_size, args.num_channels)
        self.fc2 = nn.Linear(args.num_channels, args.num_channels)
        self.fc3 = nn.Linear(args.num_channels, self.action_size)
        self.fc4 = nn.Linear(args.num_channels, 1)
        if args.cuda:
            super().cuda()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pi = self.fc3(x)    
        v = self.fc4(x)  

        return F.log_softmax(pi, dim=1), torch.tanh(v)
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
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    states, target_pis, target_vs = states.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))
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

class MCTS():
    """
    This class handles the MCTS tree.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, game, nnet, dealer_nnet, args):
        self.game = game
        self.nnet = nnet
        self.dealer_nnet = dealer_nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.get_game_ended ended for board s
        self.Vs = {}  # stores game.get_valid_actions for board s

    def get_action_prob(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(state)

        s = self.game.state_to_string(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state
        """

        s = self.game.state_to_string(state)
        current_player = state[2]

        if s not in self.Ps:
            # leaf node
            state_np = self.game.to_state_np(state)
            state_in = state_np[0]
            if current_player == 1:
                self.Ps[s], v = self.nnet.predict(state_in)
            else:
                self.Ps[s], v = self.dealer_nnet.predict(state_in)
            valids = self.game.get_valid_actions(state, current_player)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                logging.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v*current_player

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(state, current_player, a)

        if next_s[3] != 0:  #ended
            v = next_s[3]
        else:
            v = self.search(next_s)
        v *= current_player

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        state = self.game.get_init_state()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()
        ended = False
        while not ended:
            it += 1
            if verbose:
                assert self.display
                logging.debug("Turn ", str(it), "Player ", str(current_player))
                self.display(state)
            action = players[current_player + 1](state)

            valids = self.game.get_valid_actions(state, current_player)

            if valids[action] == 0:
                logging.error(f'Action {action} is not valid!')
                logging.debug(f'valids = {valids}')
                assert valids[action] > 0

            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            ended = state[3]!=0

            # Notifying the opponent for the move
            opponent = players[-current_player + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(state, action)


        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            logging.debug("Game over: Turn ", str(it), "Result ", str(state[3]))
            self.display(state)
            res = state[3]
            if res==1:
                print("Player won!")
            elif res==-1:
                print("Dealer won!")
            else:
                print("It is a tie")
        return state[3]

    def eval_game(self, start_state, player_func):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        current_player = 1
        state = start_state

        ended = False
        while not ended:
            action = player_func(state)

            valids = self.game.get_valid_actions(state, current_player)

            if valids[action] == 0:
                logging.error(f'Action {action} is not valid!')
                logging.debug(f'valids = {valids}')
                assert valids[action] > 0

            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            ended = state[3]!=0

        return state[3]

    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.play_game(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

    def eval_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.eval_games"):
            start_state = self.game.get_init_state()
            result1 = self.eval_game(start_state, self.player1)
            result2 = self.eval_game(start_state, self.player2)
            if result1 > result2:
                oneWon += 1
            elif result1 < result2:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


class Agent():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, game, nnet, args, nnargs=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, nnargs)  # the competitor network
        self.dealer_nnet = self.nnet.__class__(self.game, nnargs)  # the dealer network
        self.dealer_pnet = self.nnet.__class__(self.game, nnargs)  # the previous dealer network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.dealer_nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in load_train_examples()

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (state, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        state = self.game.get_init_state()
        current_player = state[2]
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.get_action_prob(state, temp=temp)
            state_np = self.game.to_state_np(state)
            if current_player == 1:
                trainExamples.append([state[0], current_player, pi])
                trainExamples.append([state[1], -current_player, pi])
            else:
                trainExamples.append([state[1], current_player, pi])
                trainExamples.append([state[0], -current_player, pi])

            action = np.random.choice(len(pi), p=pi)
            state= self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            r = state[3]

            if r != 0:
                return [(x[0], x[2], r *x[1], x[1]) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            logging.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for j in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.dealer_nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.execute_episode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                logging.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            #self.save_train_examples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.dealer_nnet.save_checkpoint(folder=self.args.checkpoint, filename='tempd.pth.tar')
            self.dealer_pnet.load_checkpoint(folder=self.args.checkpoint, filename='tempd.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.dealer_pnet, self.args)

            player_examples = [(x[0],x[1],x[2]) for x in trainExamples if x[3]==1]
            dealer_examples = [(x[0],x[1],x[2]) for x in trainExamples if x[3]==-1]
            self.nnet.fit(player_examples)
            self.dealer_nnet.fit(dealer_examples)

            nmcts = MCTS(self.game, self.nnet, self.dealer_nnet, self.args)

            logging.info('PLAYING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)), self.game)
            nwins, pwins, draws = arena.eval_games(self.args.games_eval)

            logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                logging.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                self.dealer_nnet.load_checkpoint(folder=self.args.checkpoint, filename='tempd.pth.tar')
            else:
                logging.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.dealer_nnet.save_checkpoint(folder=self.args.checkpoint, filename='bestd.pth.tar')
                self.save_train_examples(i - 1, best=True)

    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration, best=False):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        if best:
            filename = os.path.join(folder, "best.pth.tar.examples")
        else:
            filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def load_train_examples(self, best=False):
        if best:
            folder = self.args.checkpoint
            if not os.path.exists(folder):
                os.makedirs(folder)
            modelFile = os.path.join(folder, "best.pth.tar")
        else:
            modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            logging.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            logging.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            logging.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

args = dotdict({
    'numIters': 10,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'games_eval': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'log_level': 'INFO',
    'test': False,
    'play': False,
    'games_play': 2,
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

    game = BlackJack()

    nnet = NeuralNetModel(game, nnargs)

    if args.test:
        logging.info('Playing against self')
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        dealer_nnet = NeuralNetModel(game, nnargs)
        dealer_nnet.load_checkpoint(folder=args.checkpoint, filename='bestd.pth.tar')
        mcts = MCTS(game, nnet, dealer_nnet, args)
        arena = Arena(lambda x: np.argmax(mcts.get_action_prob(x, temp=0)),
                        lambda x: np.argmax(mcts.get_action_prob(x, temp=0)), game)
        pwins, dwins, draws = arena.play_games(args.games_eval)

        logging.info('PLAYER/DEALER WINS : %d / %d ; DRAWS : %d' % (pwins, dwins, draws))
    elif args.play:
        logging.info("Let's play!")
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        dealer_nnet = NeuralNetModel(game, nnargs)
        dealer_nnet.load_checkpoint(folder=args.checkpoint, filename='bestd.pth.tar')
        mcts = MCTS(game, nnet, dealer_nnet, args)
        cp = lambda x: np.argmax(mcts.get_action_prob(x, temp=0))
        hp = HumanBlackJackPlayer(game).play
        arena = Arena(hp, cp, game, display=BlackJack.display)
        arena.play_games(args.games_play, verbose = True)

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
    parser.add_argument("--channels", type=int, help="number of channels for the neural network",default=64)
    parser.add_argument("--games_play", type=int, help="number of games to play",default=2)
    parser.add_argument("--games_eval", type=int, help="number of games to eval",default=100)
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
    args.games_play = inargs.games_play
    args.games_eval = inargs.games_eval
    nnargs.epochs = inargs.epochs
    nnargs.num_channels = inargs.channels

    main()
