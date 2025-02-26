import logging
import pytest
from rlearn import TicTacToe, HumanTicTacToePlayer, TicTacToeModel, nnargs, BlackJack, BlackJackModel, HumanBlackJackPlayer

logging.basicConfig(level=logging.INFO)

@pytest.fixture
def setup_tictactoe_game():
    game = TicTacToe()
    model = TicTacToeModel(game, nnargs)
    human_player = HumanTicTacToePlayer(game)
    return game, model, human_player

@pytest.fixture
def setup_blackjack_game():
    game = BlackJack()
    model = BlackJackModel(game, nnargs)
    human_player = HumanBlackJackPlayer(game)
    return game, model, human_player

