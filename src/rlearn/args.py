import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import argparse
import time
import os
from tqdm import tqdm

from rlearn.utils import dotdict
from rlearn.arena import Arena
from rlearn.mcts import MCTS
from rlearn.agent import Agent
from rlearn.nnet import NeuralNetModel

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


def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Find starting indices of concatenated substrings in a string.")
    parser.add_argument("--load", action="store_true", help="load the last best checkpoint")
    parser.add_argument("--iters", type=int, help="number of iterations", default=5)
    parser.add_argument("--episodes", type=int, help="number of episodes/games for each iteration",default=1)
    parser.add_argument("--epochs", type=int, help="number of epochs for training",default=10)
    parser.add_argument("--channels", type=int, help="number of channels for the neural network",default=64)
    parser.add_argument("--loglevel", type=str, help="logging level",default='INFO')
    parser.add_argument("--eval", action="store_true", help="evaluate against self")
    parser.add_argument("--play", action="store_true", help="play against human")
    parser.add_argument("--games_play", type=int, help="number of games to play",default=2)
    parser.add_argument("--games_eval", type=int, help="number of games to eval",default=50)
    
    # Parse the arguments
    inargs = parser.parse_args()
    
    # Extract the input string and words
    args.numIters = inargs.iters
    args.numEps = inargs.episodes
    args.load_model = inargs.load
    args.log_level = inargs.loglevel
    args.eval = inargs.eval
    args.play = inargs.play
    args.games_play = inargs.games_play
    args.games_eval = inargs.games_eval

    nnargs.epochs = inargs.epochs
    if inargs.channels:
        nnargs.num_channels = inargs.channels

def run(game, nnet, human_player, agent=None):
    loglevels = dict(DEBUG=logging.DEBUG, 
                     INFO=logging.INFO,
                     WARNING=logging.WARNING,
                     ERROR=logging.ERROR,
                     CRITICAL=logging.CRITICAL 
                     )
    logging.basicConfig(level=loglevels[args.log_level])

    if args.eval:
        logging.info('Playing against self')
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        mcts = MCTS(game, nnet, args)
        arena = Arena(lambda x: np.argmax(mcts.get_action_prob(x, temp=0)),
                        lambda x: np.argmax(mcts.get_action_prob(x, temp=0)), game)
        pwins, nwins, draws = arena.play_games(args.games_eval)

        logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    elif args.play:
        logging.info("Let's play!")
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        mcts = MCTS(game, nnet, args)
        cp = lambda x: np.argmax(mcts.get_action_prob(x, temp=0))
        hp = human_player(game).play
        arena = Arena(cp, hp, game, display=game.display)
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
