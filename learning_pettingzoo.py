"""
File for me to do basic things with pettingzoo
To learn how it works

"""

from pettingzoo.butterfly import pistonball_v6
import pettingzoo

if __name__ == '__main__':
    env = pistonball_v6.env(n_pistons=20, time_penalty=-0.1, continuous=True,
    random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
    ball_elasticity=1.5, max_cycles=125, render_mode='human')
    env.reset() # resets env and sets it up for use when called the first time

    # for each agent — AEC
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # invalid action masking
            if isinstance(observation, dict) and "action_mask" in observation:
                mask = observation["action_mask"]
            else:
                mask = None

            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask=mask)

        # takes and executes the action of the agent in the environment
        # automatically switches control to next agent
        env.step(action)
    env.close()
    

    """
    AEC API usage
    env = pistonball_v6.env(n_pistons=20, time_penalty=-0.1, continuous=True,
random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
ball_elasticity=1.5, max_cycles=125, render_mode='human')
    env.reset() # resets env and sets it up for use when called the first time

    # for each agent — AEC
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # invalid action masking
            if isinstance(observation, dict) and "action_mask" in observation:
                mask = observation["action_mask"]

            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask=mask)

        # takes and executes the action of the agent in the environment
        # automatically switches control to next agent
        env.step(action)
    env.close()
    """

    """
    Parallel API usage
    env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=True,
random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
ball_elasticity=1.5, max_cycles=125, render_mode='human')
    observations, infos = parallel_env.reset(seed=42)

    while parallel_env.agents:
        # this is where you would insert your policy
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    parallel_env.close()
    """