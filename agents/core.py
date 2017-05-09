import numpy as np
from spaces import Tuple


class Actor(object):
    ''' abstraction for agents that interacts with environments '''

    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        super(Actor, self).__init__(**kwargs)

    def act(self, observation):
        raise NotImplementedError


class RandomActor(Actor):
    def act(self, observation):
        return self.action_space.sample()


class StateValueCritic(object):
    def __init__(self, observation_space):
        self.observation_space = observation_space

    def predict_v(self, observation):
        raise NotImplementedError

    def update_episode(self, trajectory):
        raise NotImplementedError

    def update_partial_episode(self, trajectory):
        raise NotImplementedError


class ActionValueCritic(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def predict_q(self, observation):
        raise NotImplementedError

    def update_episode(self, trajectory):
        raise NotImplementedError

    def update_partial_episode(self, trajectory):
        raise NotImplementedError


# class TabularRep(object):


class FirstVisitMcV(StateValueCritic):
    ''' First visit Monte Carlo method. See Sutton & Barto RL, figure 5.1. '''

    def __init__(self, observation_space, out_of_experience_func=None):
        '''
        Args:
            observation_space (Spaces) : observation space of the environment.
            out_of_experience_func (observation_space, dict(observation_space -> R) -> R) : a function that returns a state value for out-of-experience (OOE) states.
        '''
        super(FirstVisitMcV, self).__init__(observation_space)
        self.state_values = {}
        self.n_samples = {}
        self.out_of_experience_func = out_of_experience_func

    def predict_v(self, observation):
        if observation in self.state_values:
            return self.state_values[observation]
        # the observation is out-of-experience (OOE)
        if self.out_of_experience_func is None:
            raise KeyError('the observation is out-of-experience (OOE)')
        else:
            return self.out_of_experience_func(observation, self.state_values)

    def update_episode(self, observations, rewards):
        mc_returns = np.cumsum(rewards[::-1])[::-1]
        visited = set()
        for t, ob in enumerate(observations[:-1]):
            if ob not in visited:
                visited.add(ob)
                if ob not in self.state_values:
                    self.state_values[ob] = 0.
                    self.n_samples[ob] = 0
                self.n_samples[ob] += 1
                self.state_values[ob] = self.state_values[ob] + (mc_returns[t] - self.state_values[ob]) / self.n_samples[ob]


class FirstVisitMcQ(ActionValueCritic):
    ''' First visit Monte Carlo method. See Sutton & Barto RL, figure 5.1. '''
    def __init__(self, observation_space, action_space, out_of_experience_func=None):
        '''
        Args:
            observation_space (Spaces) : observation space of the environment.
            action_space (Spaces) : action space of the environment.
            out_of_experience_func (observation_space, action_space, dict(observation_space, action_space -> R) -> R) : a function that returns an action value for out-of-experience (OOE) state-action pair.
        '''
        self.every_visit_mc_v = EveryVisitMcV(
            Tuple(observation_space, action_space), out_of_experience_func)
        self.action_values = self.every_visit_mc_v.state_values
        self.out_of_experience_func = out_of_experience_func

    def predict_q(self, observation, action):
        return self.every_visit_mc_v.predict_v(observation=(observation, action))

    def update_episode(self, observations, actions, rewards):
        return self.every_visit_mc_v.update_episode(zip(observations, actions + [None]), rewards)


class EveryVisitMcV(StateValueCritic):
    ''' Every visit Monte Carlo method. See Sutton & Barto RL, chapter 5.1. '''
    def __init__(self, observation_space, out_of_experience_func=None):
        super(EveryVisitMcV, self).__init__(observation_space)
        self.state_values = {}
        self.n_samples = {}
        self.out_of_experience_func = out_of_experience_func

    def update_episode(self, observations, rewards):
        mc_returns = np.cumsum(rewards[::-1])[::-1]
        for t, ob in enumerate(observations[:-1]):
            if ob not in self.state_values:
                self.state_values[ob] = 0.
                self.n_samples[ob] = 0
            self.n_samples[ob] += 1
            self.state_values[ob] = self.state_values[ob] + (mc_returns[t] - self.state_values[ob]) / self.n_samples[ob]

    def predict_v(self, observation):
        if observation in self.state_values:
            return self.state_values[observation]
        # the observation is out-of-experience (OOE)
        if self.out_of_experience_func is None:
            raise KeyError('the observation is out-of-experience (OOE)')
        else:
            return self.out_of_experience_func(observation, self.state_values)


class EveryVisitMcQ(ActionValueCritic):
    def __init__(self, observation_space, action_space, out_of_experience_func=None):
        '''
        Args:
            observation_space (Spaces) : observation space of the environment.
            action_space (Spaces) : action space of the environment.
            out_of_experience_func (observation_space, action_space, dict(observation_space, action_space -> R) -> R) : a function that returns an action value for out-of-experience (OOE) state-action pair.
        '''
        self.first_visit_mc_v = FirstVisitMcV(
            Tuple(observation_space, action_space), out_of_experience_func)
        self.action_values = self.first_visit_mc_v.state_values
        self.out_of_experience_func = out_of_experience_func

    def predict_q(self, observation, action):
        return self.first_visit_mc_v.predict_v(observation=(observation, action))

    def update_episode(self, observations, actions, rewards):
        return self.first_visit_mc_v.update_episode(zip(observations, actions + [None]), rewards)


class TdV(StateValueCritic):
    def __init__(self, observation_space, out_of_experience_func=None):
        super(TdV, self).__init__(observation_space)
        self.state_values = {}
        self.out_of_experience_func = out_of_experience_func

    def predict_v(self, observation):
        if observation in self.state_values:
            return self.state_values[observation]
        # the observation is out-of-experience (OOE)
        if self.out_of_experience_func is None:
            raise KeyError('the observation is out-of-experience (OOE)')
        else:
            return self.out_of_experience_func(observation, self.state_values)

    # def update_partial_episode(self, observations, rewards):
    #     return self.




if __name__ == '__main__':
    import spaces
    from envs.core import MDP

    def t(s, a): return s + a if s + a < 4 else 3

    def r(s, a, sp): return 1.

    def d(s): return s == 3
    s_space = spaces.Discrete(4)
    a_space = spaces.Discrete(2)
    env = MDP(s_space, a_space, t, r, s_space.sample, d)
    actor = RandomActor(s_space, a_space)
    critic = FirstVisitMcCritic(s_space, a_space)
    ob = env.reset()
    print ob
    states, actions, rewards = [ob], [], []
    for _ in xrange(5):
        a = actor.act(ob)
        ob, r, done = env.step(a)
        states.append(ob)
        actions.append(a)
        rewards.append(r)
        print a, ob, r, done
        if done:
            critic.update((states, actions, rewards))
            states, actions, rewards = [ob], [], []
            ob = env.reset()
    print critic.action_values
    print critic.predict_q(0, 0)
    print critic.predict_q(0, 1)
