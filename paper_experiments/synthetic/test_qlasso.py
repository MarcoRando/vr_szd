import torch

from svrz.optimizers import VRSZD
from svrz.directions import QRDirections
from svrz.prox import SoftThreshold

import sys 
sys.path.append("..")

from targets import LeastSquares, QuadraticLasso
from utils import run_vrszd


d, b = 5, 3
L, mu = 100000.0, 1.0
dtype = torch.float64
device = 'cuda'


x_star = torch.ones((d, ), dtype=dtype, device=device)



#target = LeastSquares(x_star = x_star, L = L, mu = mu, seed = 12341)


target = QuadraticLasso(x_star = x_star, L = L, mu = mu, lam =1e-3, seed = 12341)


# print("SAMPLE batch: ", target.sample_z(b))
# print("FULL EVAL ON X*: ", target.full_target(x_star))

x_rnd = torch.randn((d,), dtype=target.dtype, device=target.device, generator=target.generator)

# zeta = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=target.device) #target.sample_z(b)
# print("sample: ", zeta)
# y_batch = torch.vmap(lambda z: target(x_rnd, z), in_dims=(0,))(zeta)

# print("y batch(x, z): ", y_batch)
# for (i, z) in enumerate(zeta):
#     f_i = target(x_rnd, z)
#     print(f"eval with z_i [i = {i}]", f_i.item(), "y batch[i]: ", y_batch[i].item())
#     print(torch.allclose(f_i, y_batch[i]))

# # print("full eval on x_rnd: ", target.full_target(x_rnd))
# # print("averaging f_i on all z: ", y_batch.flatten().mean(0))

# exit()
l = 5
m = 100
gamma = 1e-5
h = 1e-5

prox = SoftThreshold(lam=1e-3)
P = QRDirections(d=d, b=b, l=l, seed=12131415, dtype=dtype, device=device)
opt = VRSZD(x0=x_rnd, P=P, m=m, prox=prox, gamma=gamma, h=h, seed=12131415)

budget = 100000
vals, num_evals = run_vrszd(opt, target, budget)

