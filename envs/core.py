class MDP(object):
    ''' abstraction for Markov decision processes '''
    def __init__(self, state_space, action_space, transition_function, reward_function, sample_initial_state, termination_set):
        '''
        @param transition_function : state_space, action_space -> state_space
        @param reward_function : state_space, action_space, state_space -> R
        @param sample_initial_state : -> state_space
        @param termination_set : state_space -> bool
        '''
        self.state_space = state_space
        self.action_space = action_space
        self._transition_function = transition_function
        self._reward_function = reward_function
        self._sample_initial_state = sample_initial_state
        self._termination_set = termination_set
        self._current_state = None

    def reset(self):
        self._current_state = self._sample_initial_state()
        return self._current_state

    def step(self, action):
        assert self._current_state != None, 'MDP need to be reset first.'

        next_state = self._transition_function(self._current_state, action)
        reward = self._reward_function(self._current_state, action, next_state)
        done = self._termination_set(next_state)

        if done:
            self._current_state = None
        else:
            self._current_state = next_state

        return next_state, reward, done


class POMDP(MDP):
    ''' abstraction for partially observable Markov decision processes '''
    def __init__(self, state_space, observation_space, measurement_function, action_space, transition_function, reward_function, sample_initial_state, termination_set):
        '''
        @param measurement_function : state_space -> observation_space
        '''
        super(POMDP, self).__init__(state_space, action_space, transition_function, reward_function, sample_initial_state, termination_set)
        self.observation_space = observation_space
        self._measurement_function = measurement_function

    def reset(self):
        state = super(POMDP, self).reset()
        return self._measurement_function(state)

    def step(self, action):
        next_state, reward, done = super(POMDP, self).step(action)
        return self._measurement_function(next_state), reward, done


class AbstractPOMDP(object):
    ''' abstraction for abstract partially observable Markov decision processes, i.e. the state space is unknown '''
    def __init__(self, observation_space, action_space, reset_function, step_function):
        '''
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
    from gym import spaces
    t = lambda s, a: s + a if s + a < 4 else 3
    r = lambda s, a, sp: 1.
    d = lambda s: s == 3
    env = MDP(spaces.Discrete(4), spaces.Discrete(2), t, r, spaces.Discrete(4).sample, d)

    # print env.reset()
    # print env.step(1)
    # print env.step(1)
    # print env.step(1)

    m = lambda s: s
    pomdp = POMDP(spaces.Discrete(4), spaces.Discrete(4), m, spaces.Discrete(2), t, r, spaces.Discrete(4).sample, d)
    print pomdp.reset()
    print pomdp.step(1)
    print pomdp.step(1)
    print pomdp.step(1)
