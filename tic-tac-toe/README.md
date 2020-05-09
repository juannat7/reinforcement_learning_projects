# Tic-Tac-Toe Reinforcement Learning

Welcome to this tic-tac-toe reinforcement learning! Here you can train your own RL model and play with your trained model

## Getting Started

Clone this repository
```
git clone https://github.com/juannat95/reinforcement_learning_projects.git
cd tic-tac-toe
```

### Installing

```
pip install -r requirements.txt
```

## Training your own RL model

```
python main.py --train 1
```

You can change the training hyperparameters at configs.py. Parameters include:
- `lr (default 0.2)`: learning_rate by which the backpropagation is calculated,
- `decay_gamma (default 0.9)`: how much the reward decays per turn (ie. forces agent to win in the least possible moves),
- `exp_rate (default 0.3)`: exploration (random moves in available positions) vs exploitation (find the best possible next move from the current collection of policies learned),
- `training_epoch (default 50000)`: the number of games for the agent to train on

## Play with your trained RL model

```
python main.py --train 0
```

## Author

* **Juan Nathaniel** - *Initial work* - [Juan Nathaniel](https://github.com/juannat95)

## License

This project is licensed under the MIT License.