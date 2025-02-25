import pytest
from rlearn import TicTacToe, HumanTicTacToePlayer, TicTacToeModel, nnargs

def test_initial_state(setup_tictactoe_game):
    game, _, _ = setup_tictactoe_game
    state = game.get_init_state()
    assert state[2] == 1  # current player should be 1
    assert state[3] == 0  # reward should be 0

def test_valid_actions(setup_tictactoe_game):
    game, _, _ = setup_tictactoe_game
    state = game.get_init_state()
    valid_actions = game.get_valid_actions(state, 1)
    assert sum(valid_actions) == 9  # all positions should be valid

def test_next_state(setup_tictactoe_game):
    game, _, _ = setup_tictactoe_game
    state = game.get_init_state()
    next_state = game.get_next_state(state, 1, 0)
    assert next_state[0][0] == 1  # first position should be occupied by player 1

def test_game_ended(setup_tictactoe_game):
    game, _, _ = setup_tictactoe_game
    state = game.get_init_state()
    state[0][0] = 1
    state[0][1] = 1
    state[0][2] = 1
    assert game.get_game_ended(state, 1) == 1  # player 1 should win
