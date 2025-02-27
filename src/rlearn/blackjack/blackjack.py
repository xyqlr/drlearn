from rlearn.args import args, nnargs, parse_args, main
from rlearn.mcts import MCTS
from rlearn.blackjack import BlackJack
from rlearn.blackjack import BlackJackModel
from rlearn.blackjack import BlackJackAgent

if __name__ == "__main__":
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
        dealer_nnet.load_checkpoint(folder=args.checkpoint, filename='bestd.pth')
    else:
        agent = BlackJackAgent(game, nnet, dealer_nnet, mcts, args, nnargs)
    main(game, nnet, mcts, agent)