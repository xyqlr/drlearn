import pytest

from rlearn import BlackJack, BlackJackModel, MCTS
from rlearn import args, nnargs

def test_mcts():
    game = BlackJack()
    nnet = BlackJackModel(game, nnargs)
    dealer_nnet = BlackJackModel(game, nnargs)
    mcts = MCTS(game, nnet, dealer_nnet, args)
    state = (['J','K'], ['10', '8'], -1, 0)
    probs = mcts.get_action_prob(state) 
    