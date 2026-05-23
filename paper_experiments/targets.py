import torch 
from math import sqrt

class TargetFunction:

    def __init__(self, name : str, d : int, n : int, dtype : torch.dtype = torch.float64, device : str = 'cuda', seed : int = 189103):
        self.name = name
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.d = d 
        self.n = n
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def sample_z(self, batch_size : int):
        raise NotImplementedError("This method is implemented in subclasses!")

    def full_target(self, x):
        raise NotImplementedError("This method is implemented in subclasses!")

    def grad(self, x):
        raise NotImplementedError("This method is implemented in subclasses!")

    def __call__(self, x, z = None):
        raise NotImplementedError("This method is implemented in subclasses!")



class LeastSquares(TargetFunction):

    def __init__(self, 
            d : int,
            L : float = 10.0, 
            mu : float = 1.0,
            name : str = 'least_squares',
            dtype : torch.dtype = torch.float64,
            device : str = 'cuda',
            seed : int = 189103):
        assert L >= mu, "L must be larger or equal than mu!"
        super().__init__(name = name, d = d, n = d, dtype = dtype, device = device, seed = seed)
        self.L = L 
        self.mu = mu 
        self.A = torch.randn((self.d, self.d), dtype = self.dtype, device = self.device, generator=self.generator)
        self.x_star = torch.ones((self.d, ), dtype=self.dtype, device=self.device, requires_grad=False)

        U, S, V = self.A.svd()
        S = torch.linspace(sqrt(L), sqrt(mu), steps = self.A.shape[0], dtype = self.dtype, device = self.device, requires_grad=False)
        self.A = U @ S.diag() @ V
        self.y = self.A @ self.x_star
        self.bounds = torch.tensor([[-5.0, 5.0] for _ in range(self.d)], dtype=self.dtype, device=self.device)


    def sample_z(self, batch_size : int):
        return torch.randint(0, self.d, size=(batch_size,), device=self.device, generator=self.generator).to(dtype=torch.long)


    def __call__(self, x, z = None):
        if z is None:
            return 0.5 * (self.A @ x - self.y).norm(p=2, dim=-1).square() 

        A_z = self.A.index_select(0, z.unsqueeze(0)).squeeze(0)
        y_z = self.y.index_select(0, z.unsqueeze(0)).squeeze(0)


        return 0.5 * self.A.shape[0] * (A_z @ x - y_z).view(-1, 1).norm(p=2).square()


    def full_target(self, x):
        return self.__call__(x)

    def grad(self, x):
        return self.A.T @ (self.A @ x - self.y)


class QuadraticLasso(LeastSquares):
    def __init__(self, 
            d : int,
            L : float = 10.0, 
            mu : float = 1.0,
            lam : float = 1e-3,
            dtype : torch.dtype = torch.float64,
            device : str = 'cuda',
            seed : int = 189103):
        print("D: ", d)
        super().__init__(d = d, L = L, mu = mu, name='qlasso', dtype = dtype, device = device, seed = seed)
        self.lam = lam


    def full_target(self, x):
        fx  = self.__call__(x)
        reg = x.norm(dim=-1, keepdim=True, p=1)
        return fx + self.lam*reg