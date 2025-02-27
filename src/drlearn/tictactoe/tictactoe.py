from drlearn.args import args, nnargs, parse_args, main
from drlearn.mcts import MCTS
from drlearn.tictactoe import TicTacToe
from drlearn.tictactoe import TicTacToeModel
from drlearn.agent import Agent

if __name__ == "__main__":
    nnargs.num_channels = 64     #set the default
    args.games_sim=2
    parse_args()
    game = TicTacToe()
    nnet = TicTacToeModel(game, nnargs)
    mcts = MCTS(game, nnet, nnet, args)
    agent = None
    if not (args.eval or args.play):
        agent = Agent(game, nnet, args, nnargs)

    main(game, nnet, mcts, agent)