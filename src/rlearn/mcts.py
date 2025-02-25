import numpy as np
import logging
import math

EPS = 1e-8

class MCTS:
    """
    This class handles the MCTS tree.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, game, nnet, nnet_opponent, args):
        self.game = game
        self.nnet = nnet
        self.nnet_opponent = nnet_opponent
        self.args = args
        self.qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.nsa = {}  # stores #times edge s,a was visited
        self.ns = {}  # stores #times board s was visited
        self.ps = {}  # stores initial policy (returned by neural net)
        self.vs = {}  # stores game.get_valid_actions for board s

    def get_action_prob(self, state, temp=1):
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.num_mcts_sims):
            self.search(state)

        s = self.game.state_to_string(state)
        counts = [self.nsa[(s, a)] if (s, a) in self.nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
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
        outcome is propagated up the search path. The values of ns, nsa, qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state
        """

        s = self.game.state_to_string(state)
        current_player = state[2]
        # 1 for alternate turn
        player = 1 if self.game.alternate_turn else current_player
        v = self.game.get_game_ended(state, player)
        if v != 0:
            # terminal node
            return -v if self.game.alternate_turn else v

        if s not in self.ps:
            # leaf node
            state_np = self.game.to_neural_state(state)
            state_in = state_np[0]
            if current_player == 1:
                self.ps[s], v = self.nnet.predict(state_in)
            else:
                self.ps[s], v = self.nnet_opponent.predict(state_in)
            valids = self.game.get_valid_actions(state, player)
            self.ps[s] = self.ps[s] * valids  # masking invalid moves
            sum_ps_s = np.sum(self.ps[s])
            if sum_ps_s > 0:
                self.ps[s] /= sum_ps_s  # renormalize
            else:
                logging.error("All valid moves were masked, doing a workaround.")
                self.ps[s] = self.ps[s] + valids
                self.ps[s] /= np.sum(self.ps[s])

            self.ns[s] = 0
            return -v if self.game.alternate_turn else v

        valids = self.game.get_valid_actions(state, player)
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.qsa:
                    u = self.qsa[(s, a)] + self.args.cpuct * self.ps[s][a] * math.sqrt(self.ns[s]) / (
                            1 + self.nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.ps[s][a] * math.sqrt(self.ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(state, player, a)
        next_player = next_s[2]
        if self.game.player_agnostic_state:
            next_s = self.game.get_player_agnostic_state(next_s, next_player)

        v = self.search(next_s)
        if not self.game.alternate_turn and current_player != next_player:
            v = -v

        if (s, a) in self.qsa:
            self.qsa[(s, a)] = (self.nsa[(s, a)] * self.qsa[(s, a)] + v) / (self.nsa[(s, a)] + 1)
            self.nsa[(s, a)] += 1

        else:
            self.qsa[(s, a)] = v
            self.nsa[(s, a)] = 1

        self.ns[s] += 1
        return -v if self.game.alternate_turn else v