from core import EpisodicMDP
from spaces import NamedDiscrete
import numpy as np

class SimpleMDP(EpisodicMDP):
    def __init__(self, states, actions, transition_graph, reward_function, initial_state_probs, is_terminal):
        assert isinstance(states, (list, tuple)), '`states` has to be a list or a tuple.'
        assert isinstance(actions, (list, tuple)), '`actions` has to be a list or a tuple.'

        state_space = NamedDiscrete(states)
        action_space = NamedDiscrete(actions)

        def transition_function(state, action):
            candidate_next_states = dict([(k[2], v) for k, v in transition_graph.iteritems() if k[:2] == (state, action)])
            # default to have self transition
            candidate_probs = candidate_next_states.values()
            if np.sum(candidate_probs) < 1.:
                assert state not in candidate_next_states.items(), 'probability (state=%s, action=%s) needs to add up to 1.' % (state, action)
                candidate_next_states[state] = 1. - np.sum(candidate_probs)
            return np.random.choice(candidate_next_states.keys(), p=candidate_next_states.values())

        sample_initial_state = lambda : np.random.choice(states, p=initial_state_probs)
        termination_set = lambda state : is_terminal[state_space.to_index(state)]

        super(SimpleMDP, self).__init__(state_space=state_space, action_space=action_space, transition_function=transition_function, reward_function=reward_function, sample_initial_state=sample_initial_state, gamma=1., termination_set=termination_set)


class RecyclingRobotMDP(SimpleMDP):
    '''
    RL book by Sutton & Barto. example 3.7 Recycling robot MDP
    '''
    def __init__(self, alpha=0.6, beta=0.9, reward_search=1., reward_wait=0.1, start_at_high=0.9):
        states = ['high', 'low']
        actions = ['search', 'wait', 'recharge']

        transition_graph = {
            ('high', 'search', 'high'): alpha,
            ('high', 'search', 'low'): 1. - alpha,
            ('low', 'search', 'high'): 1. - beta,
            ('low', 'search', 'low'): beta,
            ('high', 'wait', 'high'): 1.,
            ('high', 'wait', 'low'): 0.,
            ('low', 'wait', 'high'): 0.,
            ('low', 'wait', 'low'): 1.,
            ('low', 'recharge', 'high'): 1.,
            ('low', 'recharge', 'low'): 0.,
        }

        reward_map = {
            ('high', 'search', 'high'): reward_search,
            ('high', 'search', 'low'): reward_search,
            ('low', 'search', 'high'): -3.,
            ('low', 'search', 'low'): reward_search,
            ('high', 'wait', 'high'): reward_wait,
            ('high', 'wait', 'low'): reward_wait,
            ('low', 'wait', 'high'): reward_wait,
            ('low', 'wait', 'low'): reward_wait,
            ('low', 'recharge', 'high'): 0.,
            ('low', 'recharge', 'low'): 0.,
        }

        def reward_function(state, action, next_state):
            if (state, action, next_state) in reward_map:
                return reward_map[(state, action, next_state)]
            # zero reward for self transitions
            return 0.

        super(RecyclingRobotMDP, self).__init__(states, actions, transition_graph, reward_function, [start_at_high, 1.-start_at_high], [False, False])


class RandomWalkMDP(SimpleMDP):
    '''
    RL book by Sutton & Barto. example 6.2 Random walk. With the uniformly random policy, the state values are exactly V(A) = 1/6, V(B) = 2/6, ... , V(E) = 5/6.
    '''
    def __init__(self):
        states = ['terminal', 'A', 'B', 'C', 'D', 'E']
        is_terminal = [True, False, False, False, False, False]
        actions = ['left', 'right']

        transition_graph = {}
        for i in xrange(1, len(states)):
            transition_graph[(states[i], 'left', states[i-1])] = 1.
            transition_graph[(states[i], 'right', states[(i+1) % len(states)])] = 1.
        reward_function = lambda state, action, next_state : 1. if state == 'E' and action == 'right' else 0.
        super(RandomWalkMDP, self).__init__(states, actions, transition_graph, reward_function, [0., 1./5, 1./5, 1./5, 1./5, 1./5], is_terminal)


if __name__ == '__main__':
    e = RecyclingRobotMDP(0.6, 0.9, 1., 0.1, 0.9)
    # e = RandomWalkMDP()
    print e.reset()
    for t in xrange(100):
        a = e.action_space.sample()
        print t, a, e.step(a)
