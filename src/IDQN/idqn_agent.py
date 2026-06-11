# Debug this later

import torch
from torch import nn
import numpy as np


from pettingzoo.butterfly import pistonball_v6

from dqn import DQN
from experience_replay import ReplayMemory

import itertools
import yaml
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import argparse



DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class IDQNAgent:
    def __init__(self, hyperparameter_set):
        with open('src/IDQN/hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        self.env_make_params = hyperparameters.get('env_make_params', {})

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.window = hyperparameters['window']

        self.loss_fn = nn.MSELoss()

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def train(self):
        # Setup

        env = pistonball_v6.parallel_env(**self.env_make_params, render_mode=None)
        env.reset()

        # Make dictionaries of the agent's policy networks, target networks, optimizers, and buffers
        policy_networks = {}
        target_networks = {}
        optimizers = {}
        exp_replay_buffers = {}
        for agent in env.agents:
            num_states = env.observation_space(agent).shape
            num_actions = env.action_space(agent).n
            policy_networks[agent] = DQN(num_states, num_actions).to(device)
            target_networks[agent] = DQN(num_states, num_actions).to(device)
            target_networks[agent].load_state_dict(policy_networks[agent].state_dict())
            optimizers[agent] = torch.optim.Adam(policy_networks[agent].parameters(), lr=self.learning_rate)
            exp_replay_buffers[agent] = ReplayMemory(self.replay_memory_size)
        
        # Misc rollout-relevant things
        rewards_per_episode = []
        mean_rewards = []
        best_reward = float('-inf')

        epsilon = self.epsilon_init
        epsilon_history = [self.epsilon_init]

        step_count = 0

        last_graph_update_time = datetime.now()

        agents_list = env.possible_agents

        # Rollout
        for episode in itertools.count():
            state, _ = env.reset()
            # state is a dict
            state = {agent: torch.tensor(agent_state, dtype=torch.float, device=device) for agent, agent_state in state.items()}
            terminated = False
            truncated = False
            episode_reward = 0.0

            # While agents are active
            while env.agents:
                # loop thru to get the actions
                actions = {}
                for agent in env.agents:
                    if random.random() < epsilon:
                        action = env.action_space(agent).sample()
                        actions[agent] = action
                    else:
                        with torch.no_grad():
                            # add and remove batch dim with unsqueeze & squeeze
                            action = policy_networks[agent](state[agent].unsqueeze(dim=0)).squeeze(dim=0).argmax().item()
                            actions[agent] = action

                new_state, reward, terminated, truncated, _ = env.step(actions)
                episode_reward += sum(reward.values())/len(reward) # average across agents
                new_state = {agent: torch.tensor(new_agent_state, dtype=torch.float, device=device) for agent, new_agent_state in new_state.items()}
                reward = {agent: torch.tensor(agent_reward, dtype=torch.float, device=device) for agent, agent_reward in reward.items()}

                for agent in agents_list:
                    exp_replay_buffers[agent].append((state[agent], actions[agent], new_state[agent], reward[agent], terminated[agent]))
                    state[agent] = new_state[agent]
                
                step_count += 1

            rewards_per_episode.append(episode_reward)
            if len(rewards_per_episode) >= self.window:
                mean_reward = np.mean(rewards_per_episode[-self.window:])
            else:
                mean_reward = np.mean(rewards_per_episode)
            mean_rewards.append(mean_reward)
            
            if mean_reward > best_reward:
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Episode {episode} | New best mean reward: {mean_reward:.5f}")
                self._log(log_message)
                best_reward = mean_reward
                save_model(policy_networks, self.MODEL_FILE)
            
            if mean_reward > self.stop_on_reward:
                log_message = (f"Solved! Reward: {episode_reward}")
                self._log(log_message)
                # saving here would be redundant
                break
            
            current_time = datetime.now()
            if (current_time - last_graph_update_time) > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time
            
            for agent in agents_list:
                if len(exp_replay_buffers[agent]) > self.mini_batch_size:
                    mini_batch = exp_replay_buffers[agent].sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_networks[agent], target_networks[agent], optimizers[agent])
                    
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if step_count > self.network_sync_rate:
                for agent in agents_list:
                    target_networks[agent].load_state_dict(policy_networks[agent].state_dict())
                step_count = 0
            

    def optimize(self, mini_batch, policy_dqn, target_dqn, optimizer):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)

        # actions are ints, this stacks them to (B,) either way
        actions = torch.tensor(actions, dtype=torch.long, device=device) # long bc gather needs it

        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = self.loss_fn(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def run(self):
        env = pistonball_v6.parallel_env(**self.env_make_params, render_mode='human')
        env.reset()

        policy_networks = {}
        terminations = {}
        truncations = {}
        for agent in env.agents:
            num_states = env.observation_space(agent).shape
            num_actions = env.action_space(agent).n
            policy_networks[agent] = DQN(num_states, num_actions).to(device)
            policy_networks[agent].eval()
            terminations[agent] = False
            truncations[agent] = False
        load_model(policy_networks, self.MODEL_FILE)

        state, _ = env.reset()
        state = {agent: torch.tensor(agent_state, dtype=torch.float, device=device) for agent, agent_state in state.items()}

        for _ in itertools.count(): # loop the run infinitely for funsies
            with torch.inference_mode():
                while env.agents:
                    # get action
                    actions = {}
                    for agent in env.agents:
                        if agent in terminations and (terminations[agent] or truncations[agent]):
                            actions[agent] = None
                        else:
                            actions[agent] = policy_networks[agent](state[agent].unsqueeze(dim=0)).argmax().item() # this is ok bc batch = 1 afterwards
                    
                    state, rewards, terminations, truncations, infos = env.step(actions)
                    state = {agent: torch.tensor(agent_state, dtype=torch.float, device=device) for agent, agent_state in state.items()}
        env.close()

    def save_graph(self, mean_rewards, epsilon_history):
        plt.subplot(1, 2, 1)
        plt.xlabel('Episodes')
        plt.ylabel('Mean rewards')
        plt.plot(mean_rewards)

        plt.subplot(1, 2, 2)
        plt.xlabel('Time steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        plt.savefig(self.GRAPH_FILE)
        plt.close()

    def _log(self, msg):
        print(msg)
        with open(self.LOG_FILE, 'a') as f:
            f.write(msg + '\n')
    
def save_model(policy_networks, filepath="best.pt"):
    checkpoint = {}
    for agent, network in policy_networks.items():
        checkpoint[f"{agent}_state"] = network.state_dict()
    torch.save(checkpoint, filepath)

def load_model(policy_networks, filepath="best.pt"):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        for agent, network in policy_networks.items():
            network.load_state_dict(checkpoint[f"{agent}_state"])

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    idqn = IDQNAgent(hyperparameter_set=args.hyperparameters)

    if args.train:
        idqn.train()
    else:
        idqn.run()