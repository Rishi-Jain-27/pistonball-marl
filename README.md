# pistonball-marl

An implementation of **Independent Deep Q-Networks (I-DQN)** on the [PettingZoo Pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/) environment, built to learn both the theory behind the algorithm and how to apply it to cooperative multi-agent settings.

In I-DQN, every agent runs its own standard DQN — its own Q-network, target network, optimizer, and replay buffer — and treats the other agents as part of the environment. There's no central controller and no explicit communication; cooperation emerges purely from each agent independently maximizing its own return.

## The environment

[Pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/) is a cooperative physics puzzle: a row of pistons must work together to move a ball to the left wall. Each piston is an agent that observes a local RGB image and chooses a discrete action (up / down / stay). Agents are rewarded for moving the ball leftward and penalized over time, so they must coordinate to succeed.

This makes it a good test bed for I-DQN: the agents are **homogeneous** (identical observation/action spaces) and the task is **strictly cooperative**.

## Algorithm details

- **Per-agent DQN** — each agent gets an independent policy network, target network, Adam optimizer, and replay buffer.
- **Convolutional Q-network** — observations are images, so the Q-network ([dqn.py](src/IDQN/dqn.py)) is a 3-layer CNN feature extractor feeding a 2-layer MLP head that outputs Q-values per action. Inputs are permuted to `NCHW` and normalized by 255.
- **ε-greedy exploration** — a single decaying epsilon is shared across agents to balance exploration and exploitation.
- **Experience replay** — a `deque`-backed buffer ([experience_replay.py](src/IDQN/experience_replay.py)) stores `(state, action, next_state, reward, terminated)` transitions and samples uniform mini-batches.
- **Target networks** — periodically synced from the policy networks (`network_sync_rate`) to stabilize the TD target.
- **Train per step** — the env is run in PettingZoo's `parallel_env` mode (Pistonball is not turn-based), and learning happens off the accumulated buffer rather than per episode.

### Known limitations

This is a learning implementation, so a couple of textbook MARL issues are deliberately left unsolved:

- **Non-stationarity** — because every agent's policy changes simultaneously, the environment looks non-stationary from any single agent's perspective, which breaks the usual MDP convergence guarantees.
- **Scalability** — the architecture grows in complexity with the number of agents. Strong cooperation may ultimately need a more sophisticated multi-agent algorithm.

See [plan.md](src/IDQN/plan.md) for the original design notes and possible extensions (parameter sharing, double/dueling DQN, vectorized envs).

## Project structure

```
src/IDQN/
├── dqn.py                  # Convolutional DQN network
├── experience_replay.py    # Replay buffer
├── idqn_agent.py           # IDQNAgent: train / run / optimize / logging
├── hyperparameters.yml     # Env + training hyperparameter sets
├── training.ipynb          # Colab training notebook (fallback for the CLI)
├── runs/                   # Saved models, logs, graphs + get_runs.py
└── tests/                  # Unit tests for dqn and experience_replay
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python with PyTorch, NumPy, Matplotlib, PyYAML, and `pettingzoo[butterfly]`.

## Usage

Hyperparameters are grouped into named sets in [hyperparameters.yml](src/IDQN/hyperparameters.yml) (e.g. `idqnpistonball_1`), covering both the environment make-params and the training config.

Run commands from the repository root.

**Train:**

```bash
python src/IDQN/idqn_agent.py idqnpistonball_1 --train
```

Training logs progress to `runs/<set>.log`, saves the best checkpoint to `runs/<set>.pt`, and updates a reward/epsilon plot at `runs/<set>.png`. It stops automatically once the rolling mean reward crosses `stop_on_reward`.

**Watch a trained model:**

```bash
python src/IDQN/idqn_agent.py idqnpistonball_1
```

This loads `runs/<set>.pt` and renders the agents playing with `render_mode='human'`.

### Training on Colab

GPU training is intended to run on Colab. The header of [idqn_agent.py](src/IDQN/idqn_agent.py) contains a step-by-step `colab` CLI recipe (auth → upload → install → exec → download). If the CLI flow doesn't work, [training.ipynb](src/IDQN/training.ipynb) is a notebook fallback.

### Downloading the pretrained model

A trained checkpoint is published on Kaggle and can be fetched with [runs/get_runs.py](src/IDQN/runs/get_runs.py):

```bash
python src/IDQN/runs/get_runs.py
```

## Tests

```bash
python -m pytest src/IDQN/tests/
```

## License

[MIT](LICENSE)
