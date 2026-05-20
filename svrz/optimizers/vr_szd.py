import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.utils import TargetFunction
from svrz.prox import ProxOperator
from svrz.directions import DirectionGenerator
from svrz.optimizers.abs_opt import AbsOptimizer

from enum import Enum

from svrz.utils.utils import _ffd

class VRSZDPhase(Enum):

    DET_GRAD_APPROX = 0
    INNER_LOOP_ITER = 1



class VRSZD(AbsOptimizer):

    def __init__(self, 
                    x0 : Tensor, # initial guess (shape expected (1, d))
                    P : DirectionGenerator,  
                    m : int = 10, # number of inner loop iterations
                    gamma : float = 1e-3, # stepsize
                    h : float = 1e-5, # smoothing parameter
                    prox : ProxOperator | None = None, # proximal operator
                    seed : int = 12131415):
        super().__init__(x0 = x0, P = P, P_full = None, prox = prox, seed=seed)
        self.m = m
        self.h = h if isinstance(h, Callable) else lambda _ : h
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _ : gamma

        self.I = torch.eye(P.d, dtype = P.dtype, device=P.device)
        self.phase = VRSZDPhase.DET_GRAD_APPROX
        self.tau = 0
        self.k = 0
        self.x_inner = x0.clone()
    # def _approx_grad(self, f, x, z, fx, h, P):
    #     return f(x + h * P, z).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(P.shape[1] / P.shape[0])
        

    def ask(self):
        h = self.h(self.tau)
        if self.phase == VRSZDPhase.DET_GRAD_APPROX:
            x_next = self.x0 + h * self.I
            return x_next, self.x0 #torch.cat([self.x0 + h * self.I, self.x0], dim=0)
        elif self.phase == VRSZDPhase.INNER_LOOP_ITER:
            self.G = self.P()
            x_inn_next = self.x_inner + h * self.G 
            x_out_next = self.x0 + h * self.G  
            return x_inn_next, x_out_next, self.x_inner, self.x0

    def tell(self, X, y):
        h = self.h(self.tau)
        gamma = self.gamma(self.tau)
        if self.phase == VRSZDPhase.DET_GRAD_APPROX:
            y_next, y_curr = y
 #           print(y_next.shape)
            self.g_tau = _ffd(y_next.view(1, -1), y_curr, self.I, h, nrm_const = self.x0.shape[-1])
            self.phase = VRSZDPhase.INNER_LOOP_ITER
        elif self.phase == VRSZDPhase.INNER_LOOP_ITER:
            #        values = [f_inn_next, f_out_next, f_inn_curr, f_out_curr]
            y_in_next, y_out_next, y_in_cur, y_out_cur = y

            g_k_in = _ffd(y_in_next, y_in_cur, self.G, h, nrm_const = self.x0.shape[-1])
            g_k_out = _ffd(y_out_next, y_out_cur, self.G, h, nrm_const = self.x0.shape[-1])
#            print(g_k_in)
            self.x_inner = self.prox(self.x_inner - gamma* (g_k_in - g_k_out + self.g_tau), gamma)
            self.k += 1
#            print(f"\t[--] k: {self.k}")
            if self.k >= self.m:
                self.k = 0
                self.tau += 1
                self.x0 = self.x_inner.clone()
                self.phase = VRSZDPhase.DET_GRAD_APPROX

    # def optimize(self, 
    #              f : TargetFunction,  # objective function
    #              x0: Tensor,  # initial guess
    #              T: int, # number of outer iterations
    #              m : int, # number of inner iterations
    #              gamma : float, # stepsize 
    #              h : Callable[[int], float], # smoothing parameter
    #              callback : Callable[[Tensor, float, int], None] | None = None # callback
    #              ) -> Dict:
    #     callback = callback if callback is not None else lambda x,t,iter:None

    #     x_tau = x0.clone()
    #     f_tau = f(x0).flatten().item()
    #     callback(x_tau, 0.0, 0)

    #     for tau in range(T):
    #         iteration_time = time()
    #         h_tau = h(tau)
    #         g_full = self._approx_grad(f, x_tau, None, f_tau, h_tau, self.I)
    #         x_k = x_tau.clone()
    #         for k in range(m):
    #             z_k = f.sample_z(self.batch_size)
    #             P_k = self.P()
    #             g_tau = self._approx_grad(f, x_tau, z_k, f(x_tau, z_k).flatten().item(), h_tau, P_k)
    #             g_k = self._approx_grad(f, x_k, z_k, f(x_k, z_k).flatten().item(), h_tau, P_k)
    #             x_k = self.prox(x_k - gamma * (g_k - g_tau + g_full), gamma)
    #         x_tau = x_k
    #         f_tau = f(x_tau).flatten().item()
    #         iteration_time = time() - iteration_time
    #         callback(x_tau, iteration_time, tau + 1)

    #     return x_tau 

