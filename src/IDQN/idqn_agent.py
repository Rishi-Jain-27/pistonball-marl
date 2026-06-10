
"""
imports
date format
runs directory
device = cpu or cuda if cuda is available
"""

class IDQNAgent:
    def __init__(self, hyperparameters):
        """
        open hyperparameters, init them all
        also init log model and graph dir
        """
        pass

    def train(self):
        """
        Create the parallel env

        for each agent:
        create a q-network
        target network (loaded from Q network)
        optimizer (adam)
        experience replay buffer

        create rewards per episode
        and mean rewards list
        create epsilon history
        create step count tracker and best reward is -inf

        initialize epsilon to hyperparam epsilon init

        for each episode in itertools.count() inf loop
        reset the env — get the state as a tensor

        set terminated and truncated to False
        set episode reward to 0

        while parallel_env.agents:
        for agent in env.agents:
        if random.random is below epsilon, sample an action randomly for everyhting
        and turn taht actoin to a tensor
        else just get the action from policy dqn and no grad

        out of the for gent loop, 
        get the new state, reward, terminated, truncated, info for env.step of the actoin
        accumulate episode reward with reward as a float
        convert new state and reward to tensors

        save expeience to memory - state, action, new state, reward, terminated all as a tuple
        increment step
        and make the state = new state
        
        outside of while parallel env agents:
        append episode reward to rewards per episode
        and get the mean reward
        given window thing setting if window is 100

        if episode reward is greater than best
        update best reward, save model, log

        update the graph every 10 seconds

        if the lenght of memory is greater than minibatch size
        for each agent:
        sample a minibatch from memory
        optimize miibatch, polciy, and target dqn

        epsilon is max of epsilon decay * current epsilon and the min
        append epsilon to epsiln hsitory
        still inside this loop, if the step count is > than sync rate
        copythe policy dqn to target
        set step count to 0

        """
        pass

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Note: check the tensor shape math here
        get states, actions, new states, rewads, and terminations from zip of mini batch

        do torch.stack onall of them except terminations
        for terminations make it a tensor, then .float(), then .to(device) ?

        wiht no grad:
        target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        find current q from policy network

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        find loss using self.loss_fn (MSE loss)

        optimize the model with self.optimizer
        zero grad
        loss backward
        step
        """
        pass

    def run(self):
        """
        create the parallel environment

        create the policy network
        load it
        .eval it

        reset it

        with inference mode
        while parallel_env.agents:
            insert the policy to get the actoin
            for each agent we create a dictoinary of the actions
            then parallel_env.step()
        
        parallel_env.close()

        """
        pass

    def save_graph(self, mean_rewards, epsilon_history):
        """
        We plot both epsilon decay and the mean rewards stuff

        create episodes vs mean rewards plot

        create time steps vs epsilon decay plot

        save plots to graph file
        close plots

        """
        pass

    def _log(self, msg):
        print(msg)
        with open(self.LOG_FILE, 'a') as f:
            f.write(msg + '\n')

if __name__ == '__main__':
    """
    argparsing for --train and the hyperparameter sets
    """
    pass