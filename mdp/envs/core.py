class MDP(object):
    ''' abstraction for Markov decision processes '''

    def __init__(self, state_space, action_space, transition_function, reward_function, sample_initial_state, gamma, **kwargs):
        '''
        @param state_space : Space
        @param action_space : Space
        @param transition_function : state_space, action_space -> state_space
        @param reward_function : state_space, action_space, state_space -> R
        @param sample_initial_state : -> state_space
        @param gamma : R. reward discount
        '''
        super(MDP, self).__init__(**kwargs)
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self._transition_function = transition_function
        self._reward_function = reward_function
        self._sample_initial_state = sample_initial_state
        self._current_state = None

    def reset(self):
        self._current_state = self._sample_initial_state()
        return self._current_state

    def step(self, action):
        assert self._current_state != None, '%s need to be reset first.' % self.__class__.__name__

        next_state = self._transition_function(self._current_state, action)
        reward = self._reward_function(self._current_state, action, next_state)
        self._current_state = next_state

        return next_state, reward


class EpisodicMDP(MDP):
    ''' MDP that always terminates '''

    def __init__(self, termination_set, **kwargs):
        '''
        @param termination_set : state_space -> bool
        '''
        super(EpisodicMDP, self).__init__(**kwargs)
        self._termination_set = termination_set

    def step(self, action):
        next_state, reward = super(EpisodicMDP, self).step(action)
        done = self._termination_set(next_state)

        if done:
            self._current_state = None
        else:
            self._current_state = next_state

        return next_state, reward, done


class POMDP(MDP):
    ''' abstraction for partially observable Markov decision processes '''

    def __init__(self, observation_space, measurement_function, **kwargs):
        '''
        @param observation_space : Space
        @param measurement_function : state_space -> observation_space
        '''
        super(POMDP, self).__init__(**kwargs)
        self.observation_space = observation_space
        self._measurement_function = measurement_function

    def reset(self):
        state = super(POMDP, self).reset()
        return self._measurement_function(state)

    def step(self, action):
        step_ret = super(POMDP, self).step(action)
        return (self._measurement_function(step_ret[0]),) + step_ret[1:]


class EpisodicPOMDP(POMDP, EpisodicMDP):
    pass


class AbstractPOMDP(object):
    ''' abstraction for abstract partially observable Markov decision processes, i.e. the state space is unknown '''

    def __init__(self, observation_space, action_space, reset_function, step_function):
        '''
        @param observation_space : Space
        @param action_space : Space
        @param reset_function : -> observation_space
        @param step_function : action_space -> observation_space, R, bool
        '''
        self.observation_space = observation_space
        self.action_space = action_space
        self.reset_function = reset_function
        self._step_function = step_function

    def reset(self):
        return self.reset_function()

    def step(self, action):
        return self._step_function(action)


if __name__ == '__main__':
    from mdp import spaces

    def t(s, a): return s + a if s + a < 4 else 3

    def r(s, a, sp): return 1.

    def d(s): return s == 3
    env = EpisodicMDP(
        state_space=spaces.Discrete(4),
        action_space=spaces.Discrete(2),
        transition_function=t,
        reward_function=r,
        sample_initial_state=spaces.Discrete(4).sample,
        termination_set=d,
        gamma=1.,
        )

    # print(env.reset())
    # print(env.step(1))
    # print(env.step(1))
    # print(env.step(1))

    def m(s): return s
    pomdp = EpisodicPOMDP(
        state_space=spaces.Discrete(4),
        observation_space=spaces.Discrete(4),
        action_space=spaces.Discrete(2),
        transition_function=t,
        reward_function=r,
        sample_initial_state=spaces.Discrete(4).sample,
        termination_set=d,
        gamma=1.,
        measurement_function=m,
        )
    print(pomdp.reset())
    print(pomdp.step(1))
    print(pomdp.step(1))
    print(pomdp.step(1))
