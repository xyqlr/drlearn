from rlearn.args import args, nnargs, parse_args, main
from rlearn.mcts import MCTS
from rlearn.tictactoe_game import TicTacToe
from rlearn.tictactoe_model import TicTacToeModel
from rlearn.tictactoe_player import HumanTicTacToePlayer
from rlearn.agent import Agent

if __name__ == "__main__":
    nnargs.num_channels = 64     #set the default
    args.games_sim=1
    parse_args()
    game = TicTacToe()
    nnet = TicTacToeModel(game, nnargs)
    mcts = MCTS(game, nnet, nnet, args)
    human_player = HumanTicTacToePlayer(game)
    agent = None
    if not (args.eval or args.play):
        agent = Agent(game, nnet, args, nnargs)

    main(game, nnet, mcts, human_player, agent)