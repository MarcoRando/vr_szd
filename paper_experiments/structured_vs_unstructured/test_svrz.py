import torch

import numpy as np 

from svrz.optimizers import VRSZD, VRSZDPhase
from svrz.directions import QRDirections, SphericalDirections
from svrz.prox import SoftThreshold, ProxOperator


def test_target(x, z = None):
    if z is None:
        return torch.mean(x ** 2)#, dim=1, keepdim=True) #+ torch.sum(z ** 2)
#    print(x, z)
#    print(x.shape)

    X_i = x.index_select(0, z.unsqueeze(0)).squeeze(0)
#    print(X_i)
#    print(x.square().mean().shape, X_i.shape)
#    return torch.mean(x ** 2)#, dim=1, keepdim=True)
    return X_i.square()#.mean(0, keepdim=True)




T = 1000#0000
m = 20
l = 1
d = 100
b = 1

generator = torch.Generator(device='cuda').manual_seed(121451)

x0 = torch.randn((d, ), dtype=torch.float64, generator=generator, device='cuda')
prox = ProxOperator()#lam=1e-5)
P = QRDirections(d=d, l=l, b=b, seed=12131415, device='cuda', dtype=torch.float64)
#P = SphericalDirections(d=d, l=l, b=b, seed=12131415, device='cuda', dtype=torch.float64)

opt = VRSZD(x0=x0, P=P, m=m, prox=prox, gamma=0.11, h=1e-5, seed=12131415)

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
 #       print(f"[--] F(x_next) = {f_next}")
    else:
        x_inn_next, x_out_next, x_inner, x_out = population

        zeta = torch.randint(0, d, size=(b,), device=x0.device, generator=generator).to(dtype=torch.long) #.repeat_interleave(l)#, generator=generator)

        f_inn_next = f_map(x_inn_next, zeta)
        f_out_next = f_map(x_out_next, zeta)

        f_inn_curr  = torch.vmap(lambda z : test_target(x_inner, z), in_dims=(0,))(zeta)
        f_out_curr  = torch.vmap(lambda z : test_target(x_out, z), in_dims=(0, ))(zeta) # f_map(x_out, zeta)
#        print([x.item() for x in f_inn_curr.flatten()], f_out_curr)
#        print("FIN: ", f_inn_curr)
#        exit()
        values = [f_inn_next, f_out_next, f_inn_curr, f_out_curr]
#        print("Theoretical: ",2 * m * b * (l + 1), "Pratical: ", np.sum([x.flatten().shape[0] for x in values]) )
        num_evals += np.sum([x.flatten().shape[0] for x in values])
#        exit()

#        exit()
    opt.tell(population, values)
