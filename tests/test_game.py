import pytest

from rlearn.blackjack import BlackJack, ACTION_HIT, ACTION_STAND, PLAYER, DEALER
from rlearn.tictactoe import TicTacToe

def test_blackjack_game():
    game = BlackJack()
    state = (['J','K', '2'], ['10', '8'], PLAYER, -1)
    assert  game.get_game_ended(state, PLAYER) == -1

    state = (['4', '7'], ['4', '5'], DEALER, 0)
    valids = game.get_valid_actions(state, DEALER)
    assert  valids[ACTION_STAND] == 0 and valids[ACTION_HIT] == 1

    state = game.get_init_state()
    next_state = game.get_next_state(state, 1, ACTION_STAND)
    assert next_state[0] == state[1] and next_state[1] == state[0] and next_state[2] == DEALER and next_state[3] == 0

def test_tictactoe_game():
    game = TicTacToe()
    board = [1,0,-1,-1,-1,0,1,0,1]
    state = (board, board, 1, 0)
    next_state = game.get_next_state(state, 1, 7)
    assert next_state[2] == -1 and next_state[3] == 1
