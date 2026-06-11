need to create an independent dqn algorithm to beat pistonball

How do we do this?

- Create env
discrete space
Extension later: vectorized envs
extensions: dueling and double dqn
extensoin: parameter sharing
parallel envs bc this is not a turn based game
dynamic experience replay

- Create single agent DQN
Because I-DQN runs a standard DQN for each agent, the core algorithm remains the same for every independent learner.

You need to instantiate the following for each agent:
Q-Network: A neural network that maps a state vector to Q-values for all available actions.
Target Network: A slowly updated copy of the Q-Network to stabilize training.
Optimizer: e.g., Adam or RMSprop.
Experience Replay Buffer: Stores transitions \((state, action, reward, next\_state)\)

- Choose the Action Strategy (Epsilon-Greedy)
At each step, all agents independently decide their next action using an ε-greedy policy to balance exploration and exploitation.

- Core loop
Observe: Each agent i observes its current local state \(s_{i}\).
Act: Each agent selects an action \(a_{i}\) independently using ε-greedy.
Step: Apply the joint actions \(\mathbf{a} = (a_1, a_2, \dots)\) to the environment.
Reward & Next State: The environment returns individual rewards \(r_{i}\) and new local states \(s_{i}^{\prime }\).
Store: Save the transition \((s_i, a_i, r_i, s_i')\) in Agent i's Experience Replay Buffer.
Learn: Periodically sample a mini-batch from the buffer and calculate the Loss (typically MSE or Huber Loss).
The standard single-agent target formula applies for each agent

Practical Considerations for I-DQN
Non-Stationarity: Because every agent’s policy is changing simultaneously, the environment appears non-stationary to any single agent. This makes convergence theoretically unaligned with a traditional Markov Decision Process (MDP). — i cant rlly solve this

Parameter Sharing: To help agents learn faster and mitigate non-stationarity, it is highly recommended that homogeneous agents share neural network weights while maintaining separate replay buffers or using identical policy networks. — i can do this

Scalability: The architecture becomes increasingly complex as the number of agents grows, so you may want to handle the exploding state and action space or consider more complex multi-agent algorithms if cooperation is strictly required. — cant rlly solve this

# idqn_agent.py

* top: imports, DATE_FORMAT, RUNS_DIR, device
* IDQNAgent class
* __init__ the hyperparams + log files
* train function
* run function
* save graph function
* optimize function
* _log function
* if __name__ == '__main__': argparsing

# hyperparameters.yml

* contains all hyperparameters for this
* including env make params and training params

# dqn.py

* DQN class subclass nn.module
* __init__ and super().__init__(), then create three linear layers + ReLU between them all, and action dim output, hidden_dim=256
* forward method ofc

# experience_replay.py
* init experience replay with deque
* append, sample, override __len__

# test files

* tests of all the files, build alongside, test each method/function/class

Project structure/steps:
* Comment out steps for each file before coding anything
* code experience replay and dqn and their tests before moving on to idqn
* for everything, note down tensor shapes as they flow through the network
* debug

