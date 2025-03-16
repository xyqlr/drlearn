import numpy as np
import logging
from tqdm import tqdm
from random import shuffle
from collections import deque
from drlearn.mcts import MCTS
from drlearn.agent import Agent
from drlearn.arena import Arena

class BlackJackAgent(Agent):
    def __init__(self, game, nnet, dealer_nnet, mcts, args, nnargs=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, nnargs)
        self.dealer_nnet = dealer_nnet
        self.dealer_pnet = self.nnet.__class__(self.game, nnargs)
        self.args = args
        self.mcts = mcts
        self.train_examples_history = []
        self.skip_first_self_play = False

    def simulate_game(self):
        train_examples = []
        state = self.game.get_init_state()
        current_player = state[2]
        step = 0

        while True:
            step += 1
            temp = int(step < self.args.temp_threshold)
            pi = self.mcts.get_action_prob(state, temp=temp)
            state0, state1, _, _ = self.game.to_neural_state(state)
            train_examples.append([state0, current_player, pi])
            action = np.random.choice(len(pi), p=pi)
            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            r = state[3]
            if r != 0:
                return [(x[0], x[2], x[1], x[1]) for x in train_examples]

    def learn(self):
        for i in range(1, self.args.num_iters + 1):
            logging.info(f'Starting Iter #{i} ...')
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlen_of_queue)
                for j in tqdm(range(self.args.games_sim), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.dealer_nnet, self.args)
                    iteration_train_examples += self.simulate_game()
                self.train_examples_history.append(iteration_train_examples)
            if len(self.train_examples_history) > self.args.num_iters_for_train_examples_history:
                logging.warning(f"Removing the oldest entry in train_examples. len(train_examples_history) = {len(self.train_examples_history)}")
                self.train_examples_history.pop(0)
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)
            self.nnet.save_model()
            self.pnet.load_model()
            self.dealer_nnet.save_model(filename='dealer.cur.model')
            self.dealer_pnet.load_model(filename='dealer.cur.model')
            pmcts = MCTS(self.game, self.pnet, self.dealer_pnet, self.args)
            player_examples = [(x[0], x[1], x[2]) for x in train_examples if x[3] == 1]
            dealer_examples = [(x[0], x[1], x[2]) for x in train_examples if x[3] == -1]
            self.nnet.fit(player_examples)
            self.dealer_nnet.fit(dealer_examples)
            nmcts = MCTS(self.game, self.nnet, self.dealer_nnet, self.args)
            logging.info('PLAYING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)), self.game)
            nwins, pwins, draws = arena.eval_games(self.args.games_eval)
            if i == 1:
                self.nnet.save_model()
                self.dealer_nnet.save_model(filename='dealer.cur.model')
                self.save_train_data()
            logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.update_threshold:
                logging.info('REJECTING NEW MODEL')
                self.nnet.load_model()
                self.dealer_nnet.load_model(filename='dealer.cur.model')
            else:
                logging.info('ACCEPTING NEW MODEL')
                self.nnet.save_model()
                self.dealer_nnet.save_model(filename='dealer.cur.model')
                self.save_train_data()