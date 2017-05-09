from envs.simple import RandomWalkMDP
from agents.core import RandomActor, FirstVisitMcV, FirstVisitMcQ, EveryVisitMcV, EveryVisitMcQ

''' A simple example to test and compare policy evaluation methods. See Sutton & Barto, example 6.2 '''

env = RandomWalkMDP()
agent = RandomActor(env.state_space, env.action_space)
v_critic = EveryVisitMcV(env.state_space)
q_critic = EveryVisitMcQ(env.state_space, env.action_space)

d = True
for t in xrange(1000):
    if d:
        if t != 0:
            v_critic.update_episode(obs, rewards)
            q_critic.update_episode(obs, actions, rewards)
        obs, actions, rewards = [], [], []
        o = env.reset()
        obs.append(o)

    a = agent.act(o)
    actions.append(a)
    o, r, d = env.step(a)
    obs.append(o)
    rewards.append(r)
    print a, r, o, d

true_v = {
    'A': 1. / 6,
    'B': 2. / 6,
    'C': 3. / 6,
    'D': 4. / 6,
    'E': 5. / 6,
}
print '* state values:'
print 'state, value, true_value, difference'
for state in sorted(true_v.keys()):
    v = v_critic.predict_v(state)
    print state, v, true_v[state], v - true_v[state]

print '* action values:'
print 'state, action, value, true_value, difference'
for state in sorted(true_v.keys()):
    for action in ['left', 'right']:
        q = q_critic.predict_q(state, action)
        bonus = 1. / 6 if action == 'right' else - 1. / 6
        # true_q is the same as the true_v of the resulting state
        true_q = true_v[state] + bonus
        print state, action, q, true_q, q - true_q
