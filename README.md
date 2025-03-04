# Reinforcement Learning

![Tests](https://github.com/xyqlr/drlearn/actions/workflows/run-tests.yml/badge.svg)

## Deep Reinforcement Learning With Self-Play

Deep reinforcement learning has revolutionized board games since AlphaGo beat the best human player. AlphaGo Zero was then developed without using human knowledge except the rules of the game. The main idea in AlphaGo Zero was to train a deep neural network through self-play reinforcement learning. The input to the neural network is the board position, and the output consists of a vector of values representing the probability of each move and a scalar value representing the probability of the current player winning from the position. Each step of a game is simulated by Monte Carlo Tree Search (MCTS). The MCTS search is guided by the neural network and outputs the probabilities of playing each move. When the game terminates, the score of the game (win, loss, or tie) is then propagated back as the reward. The neural network is then updated to closely match the search probabilities and maximize the reward.

The idea in AlphaGo Zero was then extended to other games like Chess and Shogi. This shows the beauty of reinforcement learning: achieving superhuman performance without human knowledge. However, the implementation requires a lot of computational power, making it hard to replicate directly to other games like Othello or even Tic-Tac-Toe.

Thanks to [alpha-zero-general](https://github.com/suragnair/alpha-zero-general), a general framework for learning any board game was implemented. This repository hopes to further extend the general idea to card games, puzzles, or any problems that can be helped by deep reinforcement learning.

Card games like Blackjack and Poker are different from board games in several aspects. First, the state is not usually in 2D, thus requiring different kinds of neural networks. Second, the outcome of a card game is typically not deterministic. Can the same idea still work? Third, there could be different types of players (dealer and player in Blackjack) or multiple players. Therefore, different strategies will be necessary, and the framework needs to be enhanced.

This repository starts with refactoring the Tic-Tac-Toe game together with the framework. Then it illustrates the refactored framework by learning the Blackjack game.

## Refactoring

- Tic-Tac-Toe: TODO

## Blackjack

TODO