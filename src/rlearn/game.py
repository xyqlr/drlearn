'''
The Game class represents the game logic, more importantly the state, the actions, and the state transiton.
This class need be subclassed by the specific game.
'''

class Game:
    def __init__(self, player_agnostic_state=True, symmetry=True):
        self.player_agnostic_state = player_agnostic_state
        self.symmetry = symmetry

    def get_init_state(self):
        '''
        state:  (player_state, opppnent_state, current_player, reward)
            player_state: a list containing the cards of the player
            opppnent_state: a list containing the cards of the dealer
                only the first card of the dealer is displayed when playing against human 
            reward:  1    :  if current_player wins
                    -1    :  if the other player wins
                    0     :  game not finished
                    1e-4  :  game is tied
        when the player stands, it switches to the dealer. for convenicne, the state becomes
                (opppnent_state, player_state, current_player, reward)
        if the game has player agnostic state, the opponent_state is the same as the player state
        '''
        pass
    
    def to_neural_state(self, state):
        '''
        input state: (player_state, opponent_state, current_player, reward)
        output state: (player_neural_state, opponent_neural_state, current_player, reward)
            a neural state is a numpy array of dimension 14, 
                the first 13 store the count of each card in a hand
                the last element is different fro the player and the opponent
                    it is the first card of the opponent for player_neural_state,
                    it is the the value of the cards of the player for opponent_neural_state.
            player_neural_state: [num As, num 2s, ..., num Ks, the first card of the opponent]
            opponent_neural_state: [num As, num 2s, ..., num Ks, the total value of the player]
        if the game has player agnostic state, the opponent_state is the same as the player state
        '''
        pass

    def get_shape(self):
        pass

    def get_action_size(self):
        '''
        the number of actions denoted as 0 to N-1, where N is the number of actions 
        '''
        pass

    def get_next_state(self, state, player, action):
        '''
        this is the critical API which controls the state transition of the game.
        if player takes action on state, return the next state
        action must be a valid move
        '''
        pass

    def get_valid_actions(self, state, player):
        '''
        given the current state, return the valid vector of actions
        '''
        pass

    def get_player_agnostic_state(self, state, player):
        '''
        if the game does not have player agnositic state, just return the state back
        '''
        pass

    def get_symmetries(self, state, pi):
        '''
        if the game does not have symmetries as most board games, just return the state back
        '''
        pass

    def get_game_ended(self, state, player):
        '''
        this returns the ending status of the game:
            1   : if the player wins
            -1  : if the player loses
            0   : a tie
            1e-4: game not ended
        '''
        pass

    def state_to_string(self, state):
        '''
        this return a string representation of the state, which needs be unique, as it is used as the key to the dictionaries in MCTS
        '''
        pass

    @staticmethod
    def display(state):
        '''
        this shows the state for playing the game against a human player
        '''
        pass

