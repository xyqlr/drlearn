import logging as log
import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import deque
from pickle import Pickler, Unpickler
import os
import sys

from drlearn.mcts import MCTS
from drlearn.arena import Arena

class Agent:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, game, nnet, args, nnargs=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, nnargs)  # the competitor network
        self.args = args
        self.train_examples_history = []  # history of examples from args.num_iters_for_train_examples_history latest iterations
        self.skip_first_self_play = False  # can be overridden in load_train_data()

    def simulate_game(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episode_step < temp_threshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form (canonical_board, curr_player, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        state = self.game.get_init_state()
        last_player = -1
        current_player = 1
        step = 0

        while True:
            step += 1
            state_ag = self.game.get_player_agnostic_state(state, current_player)
            temp = int(step < self.args.temp_threshold)

            pi = self.mcts.get_action_prob(state_ag, temp=temp)
            sym = self.game.get_symmetries(state_ag, pi)
            for b, p in sym:
                train_examples.append([b, current_player, p])

            action = np.random.choice(len(pi), p=pi)
            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            r = self.game.get_game_ended(state, current_player)
            if r != 0:
                return [(x[0], x[2], r * x[1]) for x in train_examples]

    def learn(self):
        """
        Performs num_iters iterations with games_sim episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlen_of_queue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= update_threshold fraction of games.
        """

        for i in range(1, self.args.num_iters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlen_of_queue)

                for j in tqdm(range(self.args.games_sim), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.nnet, self.args)  # reset search tree
                    iteration_train_examples += self.simulate_game()

                # save the iteration examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if len(self.train_examples_history) > self.args.num_iters_for_train_examples_history:
                log.warning(f"Removing the oldest entry in train_examples. len(train_examples_history) = {len(self.train_examples_history)}")
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # self.save_train_data(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_model()
            self.pnet.load_model()
            pmcts = MCTS(self.game, self.pnet, self.pnet, self.args)

            self.nnet.fit(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.nnet, self.args)

            log.info('PLAYING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)
            pwins, nwins, draws = arena.play_games(self.args.games_eval)

            if i == 1:
                self.nnet.save_model()
                self.save_train_data()

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_model()
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_model()
                self.save_train_data()

    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth'

    def save_train_data(self, filename='cur.data'):
        folder = os.path.join(os.path.dirname(__file__), "../../saved_models")        
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, self.game.__class__.__name__+'.'+filename)
        with open(filepath, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        f.closed

    def load_train_data(self, filename='cur.data'):
        folder = os.path.join(os.path.dirname(__file__), "../../saved_models")        
        filepath = os.path.join(folder, self.game.__class__.__name__+'.'+filename)
        if not os.path.isfile(filepath):
            log.warning(f'File "{filepath}" with train_examples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with train_examples found. Loading it...")
            with open(filepath, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True