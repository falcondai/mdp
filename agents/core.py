import numpy as np
from spaces import Tuple
from util import discount


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
    def __init__(self, observation_space, discount=1., **kwargs):
        self.observation_space = observation_space
        self.discount = discount
        super(StateValueCritic, self).__init__(**kwargs)

    def _compute_returns(self, rewards):
        return discount(rewards, self.discount)

    def predict_v(self, observation):
        raise NotImplementedError

    def update_episode(self, trajectory):
        raise NotImplementedError

    def update_partial_episode(self, incomplete_trajectory):
        raise NotImplementedError


class ActionValueCritic(object):
    def __init__(self, observation_space, action_space, discount=1., **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        super(ActionValueCritic, self).__init__(**kwargs)

    def _compute_returns(self, rewards):
        return discount(rewards, self.discount)

    def predict_q(self, observation):
        raise NotImplementedError

    def update_episode(self, trajectory):
        raise NotImplementedError

    def update_partial_episode(self, incomplete_trajectory):
        raise NotImplementedError


class Model(object):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

    def predict(self, observation):
        raise NotImplementedError

    def update(self, observation, target_value):
        raise NotImplementedError


class TabularModel(Model):
    def __init__(self, out_of_experience_func=None, **kwargs):
        '''
        Args:
            out_of_experience_func (observation_space, action_space, dict(observation_space, action_space -> R) -> R) : a function that returns an action value for out-of-experience (OOE) state-action pair.
        '''

        self._values = {}
        self._n_samples = {}
        self.out_of_experience_func = out_of_experience_func
        super(TabularModel, self).__init__(**kwargs)

    def predict(self, observation):
        if observation in self._values:
            return self._values[observation]
        # the observation is out-of-experience (OOE)
        if self.out_of_experience_func is None:
            raise KeyError('the observation is out-of-experience (OOE)')
        else:
            return self.out_of_experience_func(observation, self._values)

    def update(self, observation, target_value):
        if observation not in self._values:
            self._values[observation] = 0.
            self._n_samples[observation] = 0
        self._n_samples[observation] += 1
        self._values[observation] = self._values[observation] + (target_value - self._values[observation]) / self._n_samples[observation]


class AlphaTabularModel(TabularModel):
    def __init__(self, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        self._values = {}
        super(AlphaTabularModel, self).__init__(**kwargs)

    def update(self, observation, target_value):
        if observation not in self._values:
            self._values[observation] = 0.
        self._values[observation] = self._values[observation] + self.learning_rate * (target_value - self._values[observation])


class FirstVisitMcV(Model, StateValueCritic):
    ''' First visit Monte Carlo method. See Sutton & Barto RL, figure 5.1. '''

    def __init__(self, **kwargs):
        super(FirstVisitMcV, self).__init__(**kwargs)

    def predict_v(self, observation):
        return super(FirstVisitMcV, self).predict(observation)

    def update_episode(self, observations, rewards):
        returns = self._compute_returns(rewards)
        visited = set()
        for t, ob in enumerate(observations):
            if ob not in visited:
                visited.add(ob)
                super(FirstVisitMcV, self).update(ob, returns[t])


class QFromVCritic(StateValueCritic):
    def __init__(self, observation_space, action_space, **kwargs):
        super(QFromVCritic, self).__init__(observation_space=Tuple(observation_space, action_space), **kwargs)

    def predict_q(self, observation, action):
        return super(QFromVCritic, self).predict_v((observation, action))

    def predict_v(self, observation):
        raise NotImplementedError

    def update_episode(self, observations, actions, rewards):
        return super(QFromVCritic, self).update_episode(zip(observations, actions), rewards)

    def update_partial_episode(self, observations, actions, rewards, next_observation):
        return super(QFromVCritic, self).update_partial_episode(zip(observations, actions), rewards, next_observation)


class FirstVisitMcQ(QFromVCritic, FirstVisitMcV):
    pass


class EveryVisitMcV(Model, StateValueCritic):
    ''' Every visit Monte Carlo method. See Sutton & Barto RL, chapter 5.1. '''

    def __init__(self, **kwargs):
        super(EveryVisitMcV, self).__init__(**kwargs)

    def update_episode(self, observations, rewards):
        returns = self._compute_returns(rewards)
        for t, ob in enumerate(observations):
            super(EveryVisitMcV, self).update(ob, returns[t])

    def predict_v(self, observation):
        return super(EveryVisitMcV, self).predict(observation)


class EveryVisitMcQ(QFromVCritic, EveryVisitMcV):
    pass


class TdV(Model, StateValueCritic):
    def __init__(self, **kwargs):
        super(TdV, self).__init__(**kwargs)

    def predict_v(self, observation):
        return super(TdV, self).predict(observation)

    def update_partial_episode(self, observations, rewards, next_observation):
        obs = observations + [next_observation]
        # compute TD targets
        td_targets = np.zeros(len(rewards))
        for t, (ob, reward) in enumerate(zip(obs[:-1], rewards)):
            td_targets[t] = reward + self.discount * self.predict(obs[t+1])
        # apply update
        for t, (ob, td_target) in enumerate(zip(obs[:-1], td_targets)):
            super(TdV, self).update(ob, td_target)

    def update_episode(self, observations, rewards):
        obs = observations
        # compute TD targets
        td_targets = np.zeros(len(rewards))
        for t, (ob, reward) in enumerate(zip(obs, rewards)):
            next_state_value = self.discount * self.predict(obs[t+1]) if t < len(obs) - 1 else 0.
            td_targets[t] = reward + next_state_value
        # apply update
        for t, (ob, td_target) in enumerate(zip(obs, td_targets)):
            super(TdV, self).update(ob, td_target)


class TdQ(QFromVCritic, TdV):
    pass
