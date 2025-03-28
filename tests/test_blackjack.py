import pytest
import logging
import os
from drlearn.blackjack import BlackJack, BlackJackModel
from drlearn import args, nnargs, MCTS
from drlearn.blackjack import ACTION_HIT, ACTION_STAND, PLAYER, DEALER

def test_blackjack_initial_state(setup_blackjack_game):
    game, model = setup_blackjack_game
    state = game.get_init_state()
    assert len(state[0]) == 2  # Player's hand
    assert len(state[1]) == 2  # Dealer's hand

def test_blackjack_valid_actions(setup_blackjack_game):
    game, model = setup_blackjack_game
    state = (['10', '7'], ['9', 'A'], 0, 0)
    valids = game.get_valid_actions(state, 0)
    assert valids[0] == 1  # ACTION_HIT
    assert valids[1] == 1  # ACTION_STAND

def test_blackjack_next_state(setup_blackjack_game):
    game, model = setup_blackjack_game
    state = game.get_init_state()
    next_state = game.get_next_state(state, 0, 0)  # ACTION_HIT
    assert len(next_state[0]) == 3  # Player's hand should have one more card

@pytest.mark.parametrize("state", [
    (['10', '7'], ['9', 'A'], DEALER, 0),
    (['10', '7'], ['9', '9'], DEALER, 0),
    (['10', '7'], ['9', '8'], DEALER, 0),
    (['10', '7'], ['9', '6'], DEALER, 0),
])
def test_blackjack_get_action_prob(setup_blackjack_game, state):
    game, model = setup_blackjack_game
    best_model_path = os.path.join(os.path.dirname(__file__), "../saved_models")
    nnargs.num_channels = 512
    nnet = BlackJackModel(game, nnargs)
    dealer_nnet = BlackJackModel(game, nnargs)
    dealer_nnet.load_model(filename='dealer.1.model')
    nnet.load_model(filename='1.model')
    mcts = MCTS(game, nnet, dealer_nnet, args)

    state_n = game.to_neural_state(state)
    pi, v = dealer_nnet.predict(state_n[0])
    # Format each element of the arrays
    pi_formatted = ", ".join(f"{x:.3f}" for x in pi)
    v_formatted = ", ".join(f"{x:.3f}" for x in v)
    # Log with formatting
    logging.debug(f"pi: [{pi_formatted}], v: [{v_formatted}]")
    pi = mcts.get_action_prob(state, temp=0)
    logging.debug(f"pi: {pi}")

