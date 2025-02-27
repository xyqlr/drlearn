import numpy as np
import random
import copy

from drlearn.game import Game

ACTION_HIT = 0
ACTION_STAND = 1
PLAYER = 1
DEALER = -1

class BlackJack(Game):
    def __init__(self):
        super().__init__(alternate_turn=False, player_agnostic_state=False)
        self.n = 13
        self.reset()

    def reset(self):
        self.suite = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.deck = self.suite * 4
        random.shuffle(self.deck)
        self.current_player = 1
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        self.index_map = {k: i for i, k in enumerate(self.suite)}
        self.card_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

    def get_init_state(self):
        self.reset()
        return self.player_hand, self.dealer_hand, self.current_player, 0

    def to_neural_state(self, state):
        current_player = state[2]
        player, dealer = 0, 1
        if current_player == -1:
            player, dealer = 1, 0
        player_state = np.zeros(self.n + 1, dtype=int)
        for card in state[player]:
            player_state[self.index_map[card]] += 1
        player_state[self.n] = self.card_values[self.index_map[state[dealer][0]]]
        dealer_state = np.zeros(self.n + 1, dtype=int)
        for card in state[dealer]:
            dealer_state[self.index_map[card]] += 1
        dealer_state[self.n] = max(self._get_value(player_state))
        return (player_state, dealer_state, state[2], state[3]) if current_player == 1 else (dealer_state, player_state, state[2], state[3])

    def get_shape(self):
        return self.n, self.n

    def get_action_size(self):
        return 2

    def get_next_state(self, state, player, action):
        state0 = copy.copy(state[0])
        state1 = copy.copy(state[1])
        if action == ACTION_HIT:
            card = self._deal_next_card(state)
            state0.append(card)
            state_np, _, _, _ = self.to_neural_state((state0, state1, state[2], state[3]))
            if min(self._get_value(state_np)) > 21:
                reward = -player if player == 1 else player
                return state0, state1, state[2], reward
            else:
                return state0, state1, state[2], 0
        else:
            if player == 1:
                return state1, state0, -1, 0
            dealer_state, player_state, _, _ = self.to_neural_state(state)
            dealer_sum = max(self._get_value(dealer_state))
            player_sum = max(self._get_value(player_state))
            if player_sum > dealer_sum:
                return state0, state1, state[2], -1
            elif player_sum == dealer_sum:
                return state0, state1, state[2], 1e-4
            else:
                return state0, state1, state[2], 1

    def get_valid_actions(self, state, player):
        valids = [1] * self.get_action_size()
        current_player = state[2]
        if current_player == -1:
            dealer_state, player_state, _, _ = self.to_neural_state(state)
            dealer_value = self._get_value(dealer_state)
            if max(dealer_value) < 17:
                valids[1] = 0
            elif min(dealer_value) > 21:
                valids[0] = 0
                valids[1] = 0
        else:
            player_state, dealer_state, _, _ = self.to_neural_state(state)
            player_value = self._get_value(player_state)
            if min(player_value) > 21:
                valids[0] = 0
                valids[1] = 0
        return np.array(valids)

    def get_player_agnostic_state(self, state, player):
        return state

    def get_symmetries(self, state, pi):
        return [state, pi]

    def get_game_ended(self, state, player):
        return state[3]

    def state_to_string(self, state):
        current_player = state[2]
        if current_player == 1:
            dealer_str = str(self.card_values[self.index_map[state[1][0]]])
            player_state, _, _, _ = self.to_neural_state(state)
            return ''.join([str(v) for k, v in enumerate(player_state[:-1])]) + ":" + dealer_str + ":" + str(current_player)
        else:
            dealer_state, player_state, _, _ = self.to_neural_state(state)
            dealer_value = max(self._get_value(dealer_state))
            return ''.join([str(v) for k, v in enumerate(player_state[:-1])]) + ":" + str(dealer_value) + ":" + str(current_player)

    def play(self, state):
        valid = self.get_valid_actions(state, 1)
        str = f"Enter 0 for hit\n" if valid[0] else f""
        str += f"Enter 1 for stand" if valid[1] else f""
        print(str)
        while True:
            a = input()
            a = int(a)
            if valid[a]:
                break
            else:
                print('Invalid')
        return a
    
    @staticmethod
    def display(state):
        player_state, dealer_state, current_player, _ = state
        if current_player == -1:
            dealer_state, player_state, _, _ = state
            dealer_str = ','.join(x for x in dealer_state)
        else:
            dealer_str = dealer_state[0]
        print(f"dealer: {dealer_str}")
        player_str = ','.join(x for x in player_state)
        print(f"player: {player_str}\n")

    def _deal_next_card(self, state):
        cards = {k: 4 for k in self.suite}
        for hand in range(2):
            for card in state[hand]:
                cards[card] -= 1
        keys = list(cards.keys())
        weights = list(cards.values())
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        card = random.choices(keys, weights=probabilities, k=1)
        return card[0]

    def _get_value(self, state):
        base_value = np.sum(state[:-1] * self.card_values)
        num_aces = state[0]
        if num_aces == 0 or base_value > 21:
            return [base_value]
        possible_values = [base_value + 10 * i for i in range(num_aces + 1) if base_value + 10 * i <= 21]
        return possible_values