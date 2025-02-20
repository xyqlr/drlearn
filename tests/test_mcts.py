import pytest

from rlearn.blackjack import BlackJack, BlackJackModel, BlackJackMCTS
from rlearn.args import args, nnargs

def test_mcts():
    game = BlackJack()
    nnet = BlackJackModel(game, nnargs)
    dealer_nnet = BlackJackModel(game, nnargs)
    mcts = BlackJackMCTS(game, nnet, dealer_nnet, args)
    state = (['J','A'], ['10', '8'], -1, 0)
    probs = mcts.get_action_prob(state) 