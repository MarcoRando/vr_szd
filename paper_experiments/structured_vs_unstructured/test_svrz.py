import torch
import torch.nn.functional as F


import numpy as np 

from svrz.optimizers import VRSZD, VRSZDPhase
from svrz.directions import QRDirections, SphericalDirections
from svrz.prox import SoftThreshold, ProxOperator
from svrz.utils import _ffd




def test_target(x, z = None):
    if z is None:
        return torch.mean(x ** 2)#, dim=1, keepdim=True) #+ torch.sum(z ** 2)
#    print(x, z)
#    print(x.shape)
    print(x.shape, z.shape)
    X_i = x.index_select(0, z.unsqueeze(0)).squeeze(0)
    print(X_i.shape, x.shape, z.shape)
    ris = X_i.square()
    print(ris.shape)
    exit()
#    print(X_i)
#    print(x.square().mean().shape, X_i.shape)
#    return torch.mean(x ** 2)#, dim=1, keepdim=True)
    return ris


T = 1000#0000
m = 20
l = 5
d = 5
b = 3


def stoc_grad(x, z = None):

#    grad = torch.zeros_like(x)
    X_i = x.index_select(0, z.unsqueeze(0)).squeeze(0)
    print(X_i.shape)
#    grad[z] = 2 * x[z]

    return 2 * X_i * F.one_hot(z, num_classes=d)




generator = torch.Generator(device='cuda').manual_seed(121451)

x0 = torch.randn((d, ), dtype=torch.float64, generator=generator, device='cuda')
prox = ProxOperator()#lam=1e-5)
P = QRDirections(d=d, l=l, b=b, seed=12131415, device='cuda', dtype=torch.float64)
#P = SphericalDirections(d=d, l=l, b=b, seed=12131415, device='cuda', dtype=torch.float64)

opt = VRSZD(x0=x0, P=P, m=m, prox=prox, gamma=0.11, h=1e-10, seed=12131415)

f_map = torch.vmap(torch.vmap(lambda x, z: test_target(x, z), in_dims=(0, None)), in_dims=(0,0))
budget = 10000
num_evals = 0
while num_evals < budget:
    population = opt.ask()
    if opt.phase == VRSZDPhase.DET_GRAD_APPROX:
        x_next, x0 = population
        f_next = torch.vmap(lambda x: test_target(x), in_dims=(0,))(x_next)
        f_curr = test_target(x0)
        values = [f_next, f_curr]

        print(f"[--] F(x_tau) = {f_curr.item()}")
        num_evals += d * (l + 1)

    else:
        x_inn_next, x_out_next, x_inner, x_out = population
        print(x_inn_next.shape, x_out_next.shape, x_inner.shape, x_out.shape)
        zeta = torch.randint(0, d, size=(b,), device=x0.device, generator=generator).to(dtype=torch.long) #.repeat_interleave(l)#, generator=generator)
        print("----------------------__FIINNNNNNNNNNNNNNNNNNNN")
        f_inn_next = f_map(x_inn_next, zeta)
        f_out_next = f_map(x_out_next, zeta)
        print("----------------------__FIINNNNNNNNNNNNNNNNNNNN")
       # exit()
        f_inn_curr  = torch.vmap(lambda z : test_target(x_inner, z), in_dims=(0,))(zeta)
        f_out_curr  = torch.vmap(lambda z : test_target(x_out, z), in_dims=(0, ))(zeta) # f_map(x_out, zeta)

        values = [f_inn_next, f_out_next, f_inn_curr, f_out_curr]
        print(f_inn_next.shape, f_inn_curr.shape, x_inner.shape)
        exit()

        num_evals += np.sum([x.flatten().shape[0] for x in values])
        h = opt.h(opt.tau)
        g_k_in = _ffd(f_inn_next, f_inn_curr, opt.G, h, nrm_const = x0.shape[-1])
        g_k_out = _ffd(f_out_next, f_out_curr, opt.G, h, nrm_const = x0.shape[-1])

        grad_in = torch.vmap(lambda z :stoc_grad(x_inner, z), in_dims=(0,))(zeta).mean(0)
        grad_out = torch.vmap(lambda z :stoc_grad(x_out, z), in_dims=(0,))(zeta).mean(0)


    opt.tell(population, values)
