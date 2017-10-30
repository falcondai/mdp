from __future__ import print_function, absolute_import, division

from mdp.envs.simple import RandomWalkMDP
from mdp.agents.core import RandomActor, FirstVisitMcV, FirstVisitMcQ, EveryVisitMcV, EveryVisitMcQ, AlphaTabularModel, TabularModel, QFromVCritic, TdV, TdQ
from mdp.util import default_ooe_value, random_ooe_value

import sys
import numpy as np
import matplotlib.pyplot as plt

''' A simple example to test and compare policy evaluation methods. See Sutton & Barto, example 6.2 '''

# this environment combining with a uniformly random policy gives the same random walk transition probabilities as Sutton & Barto, example 6.2
env = RandomWalkMDP()
agent = RandomActor(env.state_space, env.action_space)
true_v = {
    'A': 1. / 6,
    'B': 2. / 6,
    'C': 3. / 6,
    'D': 4. / 6,
    'E': 5. / 6,
}

def diff_v(v_critic):
    diffs = []
    for state in sorted(true_v.keys()):
        v = v_critic.predict_v(state)
        diffs.append(true_v[state] - v)
    return np.mean(diffs)

def diff_q(q_critic):
    diffs = []
    for state in sorted(true_v.keys()):
        for action in ['left', 'right']:
            q = q_critic.predict_q(state, action)
            bonus = 1. / 6 if action == 'right' else - 1. / 6
            # true_q is the same as the true_v of the resulting state
            true_q = true_v[state] + bonus
            diffs.append(true_q - q)
    return np.mean(diffs)

# First visit MC with tables
class FvMcTabV(FirstVisitMcV, TabularModel):
    pass
class FvMcTabQ(FirstVisitMcQ, TabularModel):
    pass

mc_fv_v_critic = FvMcTabV(observation_space=env.state_space, out_of_experience_func=default_ooe_value(0.))
mc_fv_q_critic = FvMcTabQ(observation_space=env.state_space, action_space=env.action_space, out_of_experience_func=default_ooe_value(0.))


class EvMcTabV(EveryVisitMcV, TabularModel):
    pass
class EvMcTabQ(EveryVisitMcQ, TabularModel):
    pass

mc_ev_v_critic = EvMcTabV(observation_space=env.state_space, out_of_experience_func=default_ooe_value(0.))
mc_ev_q_critic = EvMcTabQ(observation_space=env.state_space, action_space=env.action_space, out_of_experience_func=default_ooe_value(0.))


class TdTabV(TdV, AlphaTabularModel):
    pass
class TdTabQ(TdQ, AlphaTabularModel):
    pass

td_v_critic = TdTabV(observation_space=env.state_space, out_of_experience_func=default_ooe_value(0.), learning_rate=0.05)
td_q_critic = TdTabQ(observation_space=env.state_space, action_space=env.action_space, out_of_experience_func=default_ooe_value(0.), learning_rate=0.05)

# XXX initialization of V/Q are very important (via the OOE function)
# online_td_v_critic = TdTabV(observation_space=env.state_space, out_of_experience_func=random_ooe_value(-1., 1.), learning_rate=0.05)
online_td_v_critic = TdTabV(observation_space=env.state_space, out_of_experience_func=default_ooe_value(0.), learning_rate=0.05)
online_td_q_critic = TdTabQ(observation_space=env.state_space, action_space=env.action_space, out_of_experience_func=default_ooe_value(0.), learning_rate=0.05)

v_critics = [mc_fv_v_critic, mc_ev_v_critic, td_v_critic]
q_critics = [mc_fv_q_critic, mc_ev_q_critic, td_q_critic]
labels = ['first-visit MC %s', 'every-visit MC %s', 'batch TD(0) %s']

ticks = [0]
mean_diff_vs = [[diff_v(v_critic)] for v_critic in v_critics]
mean_diff_qs = [[diff_q(q_critic)] for q_critic in q_critics]

online_mean_diff_vs = []
online_mean_diff_qs = []

d = True
n_ticks = 10000 if len(sys.argv) < 2 else int(sys.argv[1])
for t in xrange(n_ticks):
    if d:
        # update critics
        if t != 0:
            ticks.append(t)
            for i, v_critic in enumerate(v_critics):
                v_critic.update_episode(obs[:-1], rewards)
                mean_diff_vs[i].append(diff_v(v_critic))
            for i, q_critic in enumerate(q_critics):
                q_critic.update_episode(obs[:-1], actions, rewards)
                mean_diff_qs[i].append(diff_q(q_critic))

        obs, actions, rewards = [], [], []
        o = env.reset()
        obs.append(o)

    a = agent.act(o)
    actions.append(a)
    prev_o = o
    o, r, d = env.step(a)
    obs.append(o)
    rewards.append(r)

    # online TD(0)
    online_td_v_critic.update_partial_episode([prev_o], [r], o)
    online_td_q_critic.update_partial_episode([prev_o], [a], [r], (o, agent.act(o)))
    online_mean_diff_vs.append(diff_v(online_td_v_critic))
    online_mean_diff_qs.append(diff_q(online_td_q_critic))

plt.subplot(1, 2, 1)
plt.title('state values')
plt.axhline(0., c='black')
for label, vs in zip(labels, mean_diff_vs):
    plt.plot(ticks, vs, label=label % 'V')
plt.plot(range(n_ticks), online_mean_diff_vs, label='online TD(0) V')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('action values')
plt.axhline(0., c='black')
for label, qs in zip(labels, mean_diff_qs):
    plt.plot(ticks, qs, label=label % 'Q')
plt.plot(range(n_ticks), online_mean_diff_qs, label='online TD(0) Q')
plt.legend()

plt.show()
