from drlearn.args import args, nnargs, parse_args, main
from drlearn.mcts import MCTS
from drlearn.blackjack import BlackJack
from drlearn.blackjack import BlackJackModel
from drlearn.blackjack import BlackJackAgent

def run():
    nnargs.num_channels = 512
    args.num_mcts_sims = 50
    args.games_sim = 100
    parse_args()
    game = BlackJack()
    nnet = BlackJackModel(game, nnargs)
    dealer_nnet = BlackJackModel(game, nnargs)
    mcts = MCTS(game, nnet, dealer_nnet, args)
    agent = None
    if args.eval or args.play:
        dealer_nnet.load_model()
    else:
        agent = BlackJackAgent(game, nnet, dealer_nnet, mcts, args, nnargs)
    main(game, nnet, mcts, agent)

if __name__ == "__main__":
    run()