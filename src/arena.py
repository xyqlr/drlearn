import logging
from tqdm import tqdm

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        state = self.game.get_init_state()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.get_game_ended(state, current_player) == 0:
            it += 1
            if verbose:
                assert self.display
                logging.debug("Turn ", str(it), "Player ", str(current_player))
                self.display(state)
            state_ag = self.game.get_player_agnostic_state(state, current_player)
            action = players[current_player + 1](state_ag)

            valids = self.game.get_valid_actions(state_ag, 1)

            if valids[action] == 0:
                logging.error(f'Action {action} is not valid!')
                logging.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-current_player + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(state, action)

            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            logging.debug("Game over: Turn ", str(it), "Result ", str(state[3]))
            res = current_player*self.game.get_game_ended(state, current_player)
            if res==1:
                print("Player won!")
            elif res==-1:
                print("The other won!")
            else:
                print("It is a tie")
        return current_player*self.game.get_game_ended(state, current_player) 

    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            gameResult = self.play_game(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            gameResult = self.play_game(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

    def eval_game(self, start_state, player_func):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        current_player = 1
        state = start_state

        ended = False
        while not ended:
            action = player_func(state)

            valids = self.game.get_valid_actions(state, current_player)

            if valids[action] == 0:
                logging.error(f'Action {action} is not valid!')
                logging.debug(f'valids = {valids}')
                assert valids[action] > 0

            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            ended = state[3]!=0

        return state[3]

    def eval_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.eval_games"):
            start_state = self.game.get_init_state()
            result1 = self.eval_game(start_state, self.player1)
            result2 = self.eval_game(start_state, self.player2)
            if result1 > result2:
                oneWon += 1
            elif result1 < result2:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
