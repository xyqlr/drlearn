import argparse
import logging
import numpy as np

from drlearn.utils import DotDict
from drlearn.arena import Arena

args = DotDict({
    'num_iters': 10,
    'games_sim': 1,  # Number of complete self-play games to simulate during a new iteration.
    'temp_threshold': 15,
    'update_threshold': 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlen_of_queue': 200000,  # Number of game examples to train the neural networks.
    'num_mcts_sims': 25,  # Number of game moves for MCTS to simulate.
    'games_eval': 50,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'load_model': None,
    'num_iters_for_train_examples_history': 20,
    'log_level': 'INFO',
    'test': False,
    'play': False,
    'games_play': 2,
})

nnargs = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class DefaultStringAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string is None:
            # If the argument is not specified, do nothing
            return
        if values is None:
            # If the argument is specified without a value, use the default
            setattr(namespace, self.dest, self.default)
        else:
            # If the argument is specified with a value, use that value
            setattr(namespace, self.dest, values)

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, help="number of iterations")
    parser.add_argument("--games_sim", type=int, help="number of simulated games for each iteration")
    parser.add_argument("--epochs", type=int, help="number of epochs for training")
    parser.add_argument("--channels", type=int, help="number of channels for the neural network")
    parser.add_argument("--log_level", type=str, help="logging level", default='INFO')
    parser.add_argument("--eval", action="store_true", help="evaluate against self")
    parser.add_argument("--play", action="store_true", help="play against human")
    parser.add_argument("--games_play", type=int, help="number of games to play")
    parser.add_argument("--games_eval", type=int, help="number of games to eval")
    # Add the argument with a custom action
    parser.add_argument(
        '--load',
        action=DefaultStringAction,
        default=None,  # Default value if the argument is not specified
        const='cur', # Default value if the argument is specified without a value
        nargs='?',     # Allows the argument to be specified with or without a value
        help='load a saved model (default: "cur" if --load is used without a value)'
    )

    # Parse the arguments
    inargs = parser.parse_args()

    # Check if the argument --load is specified
    if inargs.load is not None:
        args.load_model = inargs.load
    if inargs.iters:
        args.num_iters = inargs.iters
    if inargs.games_sim:
        args.games_sim = inargs.games_sim
    args.log_level = inargs.log_level
    args.eval = inargs.eval
    args.play = inargs.play
    if inargs.games_play:
        args.games_play = inargs.games_play
    if inargs.games_eval:
        args.games_eval = inargs.games_eval
    if inargs.epochs:
        nnargs.epochs = inargs.epochs
    if inargs.channels:
        nnargs.num_channels = inargs.channels


def main(game, nnet, mcts, agent=None):
    loglevels = dict(DEBUG=logging.DEBUG,
                     INFO=logging.INFO,
                     WARNING=logging.WARNING,
                     ERROR=logging.ERROR,
                     CRITICAL=logging.CRITICAL)
    logging.basicConfig(level=loglevels[args.log_level])

    if args.eval:
        logging.info('Playing against self')
        nnet.load_model()
        arena = Arena(lambda x: np.argmax(mcts.get_action_prob(x, temp=0)),
                      lambda x: np.argmax(mcts.get_action_prob(x, temp=0)), game)
        pwins, nwins, draws = arena.play_games(args.games_eval)

        logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    elif args.play:
        logging.info("Let's play!")
        nnet.load_model()
        cp = lambda x: np.argmax(mcts.get_action_prob(x, temp=0))
        hp = game.play
        arena = Arena(hp, cp, game, display=game.display)
        arena.play_games(args.games_play, verbose=True)
    else:
        assert agent is not None

        if args.load_model is not None:
            logging.info('Loading model')
            nnet.load_model(filename=f"{args.load_model}.model")
        else:
            logging.warning('Not loading a model!')

        if args.load_model:
            logging.info("Loading 'trainExamples' from file...")
            agent.load_train_data(filename=f"{args.load_model}.data")

        logging.info('Starting the learning process 🎉')
        agent.learn()