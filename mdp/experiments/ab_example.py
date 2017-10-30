from mdp.spaces import NamedDiscrete

observation_space = NamedDiscrete(['A', 'B', 'terminal1', 'terminal2'])

episodes = [
    ['A', 0, 'B', 0],
    ['B', 1],
    ['B', 1],
    ['B', 1],
    ['B', 1],
    ['B', 1],
    ['B', 1],
    ['B', 0],
]
