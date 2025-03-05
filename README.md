![Tests](https://github.com/xyqlr/drlearn/actions/workflows/run-tests.yml/badge.svg)

# Deep Reinforcement Learning With Self-Play

Deep reinforcement learning has revolutionized board games since [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) beat the best human player. [AlphaGo Zero](https://www.semanticscholar.org/paper/Mastering-the-game-of-Go-without-human-knowledge-Silver-Schrittwieser/c27db32efa8137cbf654902f8f728f338e55cd1c) was then developed without human knowledge except the rules of the game. The main idea in [AlphaGo Zero](https://www.semanticscholar.org/paper/Mastering-the-game-of-Go-without-human-knowledge-Silver-Schrittwieser/c27db32efa8137cbf654902f8f728f338e55cd1c) was to train a deep neural network through self-play reinforcement learning. The input to the neural network is the board position, and the output consists of a vector of values representing the probability of each move and a scalar value representing the probability of the current player winning from the position. Each step of a game is simulated by Monte Carlo Tree Search (MCTS). The MCTS search is guided by the neural network and outputs the probabilities of playing each move. When the game terminates, the score of the game (win, loss, or tie) is then propagated back as the reward. The neural network is then updated to closely match the search probabilities and maximize the reward.

The idea in AlphaGo Zero was then extended to other games like Chess and Shogi, leading to the development of [AlphaZero](https://arxiv.org/abs/1712.01815). This shows the beauty of reinforcement learning: achieving superhuman performance without human knowledge, while supervised learning is limited by human knowledge. However, the implementation requires a lot of computational power, making it hard to replicate directly to other games like Othello or even Tic-Tac-Toe.

Thanks to [alpha-zero-general](https://github.com/suragnair/alpha-zero-general), a general framework for learning any board game was implemented. This repository hopes to further extend the general idea to card games, puzzles, or any problems that can be helped by deep reinforcement learning.

Card games like Blackjack and Poker are different from board games in several aspects. First, the state is not usually in 2D, thus requiring different kinds of neural networks. Second, the outcome of a card game is typically not deterministic. Can the same idea still work? Third, there could be different types of players (dealer and player in Blackjack) or multiple players. Therefore, different strategies will be necessary, and the framework needs to be enhanced. For example, the evaluation is very different for Blackjack, as the player can only play against the dealer instead of against itself.

This repository starts with refactoring the Tic-Tac-Toe game along with the framework. Then it illustrates the refactored framework by learning the Blackjack game.

## Repository Structure

- The code is organized as a package named drlearn.
- Each game is under a subpackage under package drlearn, and has its own main method.
- MCTS is added an opponent model for the case that the state is not player agnostic.
- The play method for the human player of a game is merged to the game class.
- The best_models subfolder archives the best model and training data for a game.
- The tests subfolder has the unit tests, which are run in the GitHub workflow.

## Tic-Tac-Toe

To see the usage:
```bash
python src/drlearn/tictactoe/tictactoe.py --help
```

To train the Tic-Tac-Toe model from scratch:
```bash
python src/drlearn/tictactoe/tictactoe.py
```

To load a previously saved model and continue training:
```bash
python src/drlearn/tictactoe/tictactoe.py --load
```

To evaluate the saved model against itself:
```bash
python src/drlearn/tictactoe/tictactoe.py --eval
```

To play against the trained model:
```bash
python src/drlearn/tictactoe/tictactoe.py --play
```

## Blackjack

### Rules

Blackjack is a card game where players try to get a hand value as close to 21 as possible without exceeding it. The game is played with one or more decks of 52 cards. Each card has a value: numbered cards are worth their face value, face cards (Jack, Queen, King) are worth 10, and Aces can be worth 1 or 11.

- Each player is dealt two cards, and the dealer is dealt two cards (one face up, one face down).
- Players can choose to "hit" (take another card) or "stand" (keep their current hand).
- If a player's hand exceeds 21, they "bust" and lose the game.
- After all players have finished their turns, the dealer reveals their face-down card and must hit until their hand is 17 or higher.
- The player with a hand value closest to 21 without exceeding it wins. If the dealer busts, any remaining players win.

What's shown here is one player playing against the dealer with one deck of cards.

There is a Blackjack environment in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#blackjack). The environment simulates the Blackjack game and can be used to train reinforcement learning agents. Why do we still develop the Blackjack game here?

The main reason is that we want to extend the general framework to card games, and Blackjack is simple enough to start with. Another reason is that there is no model for the dealer in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#blackjack), which could be problematic when there are multiple players. Even with one player, it seems to assume some probability distributions of the cards in the dealer's default decision. Will the dealer always stand after reaching 17 or higher and having equal value to the player?

We argue that the dealer can still make a choice between hit or stand even in this situation. The decision may turn out to be trivial, but we'd like the model to learn it. Therefore, we train a separate model for the dealer, which does not have the player agnostic state, and needs different training data.

### Observation State vs Neural State

The state in the Blackjack environment in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#blackjack) consists of three components:
1. The player's current hand value.
2. The dealer's visible card.
3. A boolean indicating whether the player has a usable Ace (an Ace that can be counted as 11 without busting).

We argue that the state has encoded some non-trivial human knowledge: the hand value of the player, and whether an Ace is counted as 1 or 11. Therefore, we propose to use a different representation for the state with the goal that the model should see the raw input if possible. We also distinguish the observation state and the neural state for the player and the dealer.

The observation state for the player consists of the cards in the player's hand plus the dealer's visible card. The observation state for the dealer consists of the cards in the dealer's hand plus the player's cards.

The observation states cannot be fed to the neural network models, because they are characters of different lengths. They are converted to the neural states, which are vectors of the same length. For the player, the neural state consists of the frequency of each card (ignoring the suits) in the player's hand plus the dealer's visible card index (with value 1 to 13). For the dealer, the neural state consists of the frequency of each card in the dealer's hand plus the player's maximal value (with Ace having value 11).

### MCTS for the Player and the Dealer

MCTS is enhanced to have an opponent model. For board games with player agnostic states like Tic-Tac-Toe, the opponent model is the same as the player's. For Blackjack, the opponent model is for the dealer, and is trained with the self-play data for the dealer only.

For board games with alternate turns, the reward is backpropagated from the terminal nodes with signs negated every time. For Blackjack, the reward is only changed signs when switching from the dealer to the player.

### Evaluation Against The Previous Model

For board games like Tic-Tac-Toe, we can evaluate the new model against an old model by letting them play against each other. This, however, does not work for Blackjack, because two players cannot play against each other. Note that the player can only play against the dealer. We propose a new method for evaluating a new model against an old one.

Given an initial state (the player has two cards, and the dealer has two but only the first one visible), we first let the new player model play against the latest dealer model. Then we let the old player model play against the latest dealer model. If the new player model wins and the old player model loses or ties, or the new player model ties and the old player model loses, then we say the new player model wins the game. Accordingly, we can decide whether the new player model ties or loses the game. At the end, if the new player model wins over some preset margin, we replace the old player model with the new player model.

### Experimental Results

