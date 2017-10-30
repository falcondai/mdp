import numpy as np


class Space(object):
    def sample(self):
        raise NotImplementedError


class Discrete(Space):
    def __init__(self, cardinality):
        self.cardinality = cardinality

    def sample(self):
        return np.random.randint(self.cardinality)


class NamedDiscrete(Discrete):
    def __init__(self, names):
        assert isinstance(names, (list, tuple)
                          ), '`names` must be a list or a tuple.'
        self.cardinality = len(names)
        self._name_to_index = dict(zip(names, range(self.cardinality)))
        self._index_to_name = dict(zip(range(self.cardinality), names))
        self._names = names

    @property
    def name(self):
        return self._names

    def sample(self):
        return np.random.choice(self._names)

    def to_index(self, name):
        return self._name_to_index[name]

    def to_name(self, index):
        return self._index_to_name[index]


class Box(Space):
    def __init__(self, low, high):
        assert np.shape(low) == np.shape(
            high), 'the shapes of `low` and `high` must be equal.'
        if np.shape(low) == ():
            # takes care of the case where low and high are scalar
            low, high = [low], [high]
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.ptp = self.high - self.low
        self.shape = np.shape(low)

    def sample(self):
        return np.random.rand(*self.shape) * self.ptp + self.low


class Lattice(Space):
    def __init__(self, low, high):
        assert np.shape(low) == np.shape(
            high), 'the shapes of `low` and `high` must be equal.'
        if np.shape(low) == ():
            # takes care of the case where low and high are scalar
            low, high = [low], [high]
        self.low = np.asarray(low, dtype=np.int32)
        self.high = np.asarray(high, dtype=np.int32)
        self.ptp = self.high - self.low
        self.shape = np.shape(low)

    def sample(self):
        return np.asarray(np.random.rand(*self.shape) * self.ptp + self.low, dtype=np.int32)


class Screen(Lattice):
    def __init__(self, height, width, n_channels):
        assert isinstance(height, int), '`height` must be an integer.'
        assert isinstance(width, int), '`width` must be an integer.'
        assert isinstance(n_channels, int), '`n_channels` must be an integer.'
        shape = (height, width, n_channels)
        super(Screen, self).__init__(np.zeros(shape, dtype=np.uint8),
                                     (2**8 - 1) * np.ones(shape, dtype=np.uint8))


class Tuple(Space):
    def __init__(self, *spaces):
        assert len(spaces) > 0, '`spaces` must be non-empty.'
        for space in spaces:
            assert isinstance(space, Space), 'each space must be a `Space`.'
        self.spaces = tuple(spaces)

    def sample(self):
        return tuple([space.sample() for space in self.spaces])


class Union(Space):
    def __init__(self, spaces):
        assert len(spaces) > 0, '`spaces` must be non-empty.'
        for space in spaces:
            assert isinstance(space, Space), 'each space must be a `Space`.'
        self.spaces = tuple(spaces)

    def sample(self):
        # randomly select a subspace then sample from the subspace
        # note that this is not a uniform distribution over states
        return np.random.choice(self.spaces).sample()


if __name__ == '__main__':
    b = Lattice([0, 10], [2, 20])
    print(b.low, b.high, b.ptp, b.shape)
    print(b.sample())

    c = Discrete(100)
    t = Tuple(b, c)
    print(t.sample())

    s = Screen(2, 4, 3)
    print(s.sample())
