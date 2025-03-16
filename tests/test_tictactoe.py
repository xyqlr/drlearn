import pytest
import numpy as np
import logging
import os
from drlearn.tictactoe import TicTacToe, TicTacToeModel
from drlearn import MCTS, args, nnargs

def test_initial_state(setup_tictactoe_game):
    game, _ = setup_tictactoe_game
    state = game.get_init_state()
    assert state[2] == 1  # current player should be 1
    assert state[3] == 0  # reward should be 0

def test_valid_actions(setup_tictactoe_game):
    game, _ = setup_tictactoe_game
    state = game.get_init_state()
    valid_actions = game.get_valid_actions(state, 1)
    assert sum(valid_actions) == 9  # all positions should be valid

def test_next_state(setup_tictactoe_game):
    game, _ = setup_tictactoe_game
    state = game.get_init_state()
    next_state = game.get_next_state(state, 1, 0)
    assert next_state[0][0] == 1  # first position should be occupied by player 1

def test_game_ended(setup_tictactoe_game):
    game, _ = setup_tictactoe_game
    state = game.get_init_state()
    state[0][0] = 1
    state[0][1] = 1
    state[0][2] = 1
    assert game.get_game_ended(state, 1) == 1  # player 1 should win

@pytest.mark.parametrize("state", [
    (np.array([1,0,0,0,0,0,0,0,0]), np.array([1,0,0,0,0,0,0,0,0]), -1, 0),
    (np.array([0,0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0,0]), 1, 0),
])
def test_tictactoe_get_action_prob(setup_tictactoe_game, state):
    game, model = setup_tictactoe_game
    best_model_path = os.path.join(os.path.dirname(__file__), "../saved_models")
    nnargs.num_channels = 64
    nnet = TicTacToeModel(game, nnargs)
    nnet.load_model(filename='1.model')
    mcts = MCTS(game, nnet, nnet, args)

    state = game.get_player_agnostic_state(state, -1)
    pi, v = nnet.predict(state[0])
    # Format each element of the arrays
    pi_formatted = ", ".join(f"{x:.3f}" for x in pi)
    v_formatted = ", ".join(f"{x:.3f}" for x in v)
    # Log with formatting
    logging.debug(f"pi: [{pi_formatted}], v: [{v_formatted}]")

    pi = mcts.get_action_prob(state, temp=0)
    pi_formatted = ", ".join(f"{x:.3f}" for x in pi)
    logging.debug(f"pi: [{pi_formatted}]")

