from torch import Generator, randint


class TargetFunction:
    
    def __init__(self, n : int, seed : int = 12131415):
        assert n > 0
        self.n = n
        self.generator = Generator()
        self.generator.manual_seed(seed)
    
    def sample_z(self, i = 1):
        return randint(0, self.n, size=(i,), generator=self.generator)
    
    def __call__(self, x, z = None):
        raise NotImplementedError("Implemented in sub-classes")