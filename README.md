# Hexapawn Neural Network – Minimax Supervision & AlphaZero-Style Self Play

This project implements a neural-network-based Hexapawn agent trained using **two complementary approaches**:

1. **Supervised learning** from data generated via an **exhaustive minimax search**
2. **AlphaZero-style reinforcement learning** through **self-play combined with Monte Carlo Tree Search (MCTS)**

The implementation closely follows the ideas presented in  
**Dominik Klein – _Neural Networks for Chess: The Magic of Deep and Reinforcement Learning Revealed_**,  
with several deliberate modifications and extensions.

---

## Key Features & Design Choices

### 🔹 Game Environment
- The game of **Hexapawn** is fully implemented from scratch.
- All legal moves, terminal conditions, and win/loss states are explicitly handled.
- Board positions are represented **from the perspective of the side to move**, meaning:
  - The neural network is *blind to color*
  - The same position from opposite sides is normalized into a single representation

This removes the need for an explicit *side-to-play* input and enforces symmetry.

---

### 🔹 Training Methods

#### 1. Supervised Learning (Minimax Oracle)
- An **exhaustive minimax search** is used to compute:
  - Exact game-theoretic values
  - Optimal move targets
- These results are stored in `data/data.json`
- The neural network is trained to imitate perfect play

This produces a strong baseline model and serves as a sanity check for the learning pipeline.

#### 2. Reinforcement Learning (AlphaZero Style)
- The agent plays against itself using **Monte Carlo Tree Search**
- The neural network guides:
  - **Policy** (prior over moves)
  - **Value** (position evaluation)
- Self-play games are used to generate training data
- The network is updated iteratively over multiple training cycles

---

### 🔹 Neural Network
- Implemented in **PyTorch**

---

## Project Structure

├── data/
│   └── data.json          # Dataset from exhaustive minimax search (board states and optimal moves)
├── models/
│   └── [model files]      # Saved PyTorch models from supervised and reinforcement learning
├── Hexapawn.py            # Defines the Hexapawn game class (board representation, moves, rules)
├── model.py               # Defines the Neural Network class in PyTorch
├── minimax.py             # Implements the minimax algorithm for exhaustive search and data generation
├── mcts.py                # Implements Monte Carlo Tree Search (MCTS) and self-play for reinforcement learning
├── main.py                # Main script: Runs minimax for data generation, supervised training, self-play RL, and model evaluation
├── requirements.txt       # Shows the required installations
└── README.md              # This file


---

## How It Works

### Supervised Training Pipeline
1. Run exhaustive minimax on all reachable positions
2. Store optimal moves and position values in `data.json`
3. Train the neural network to predict:
   - Best move (policy)
   - Game outcome (value)

---

### AlphaZero-Style Training Pipeline
1. Initialize neural network (random)
2. Perform self-play games using MCTS
3. Collect `(state, policy_target, value_target)` tuples
4. Train the network on generated data
5. Repeat over multiple iterations

---

## Evaluation
- Models are evaluated by playing against random opponents with the black pieces, with perfect conversion being the required result

---

## Requirements

- Python 3.9+
- PyTorch


---

## Motivation

Hexapawn is small enough to:
- Allow **exact minimax solutions**
- Make AlphaZero-style learning **fully interpretable**
- Serve as a controlled testbed for:
  - MCTS + neural networks
  - Value/policy learning dynamics
  - Architectural and representation choices

This project is primarily educational and experimental.

---

## References

- Dominik Klein, *Neural Networks for Chess*
- AlphaZero (Silver et al., 2017)

---

