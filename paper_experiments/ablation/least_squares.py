import os
import sys
import torch

from svrz.prox import SoftThreshold

sys.path.append("../")


from targets import LeastSquares
from datasets import SyntheticDataset
from utils import get_optimizer, get_cost_per_iter, test_optimizer




opt_name = sys.argv[1]
L = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
mu = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

d = int(sys.argv[4]) 
m = int(sys.argv[5]) 
l = int(sys.argv[6]) 

out_dir = sys.argv[7]

os.makedirs(f"{out_dir}/param_tuning", exist_ok=True)
os.makedirs(f"{out_dir}/full_results", exist_ok=True)




dtype = torch.float64
device = 'cuda'
x_start = torch.zeros((d, ), dtype=dtype, device=device)

seed = 12131415
lam = 1e-5

data = SyntheticDataset(x_star=x_start,n = d, L = L, mu = mu, seed=seed)

target = LeastSquares(data=data, lam = lam, seed=seed)

prox = SoftThreshold(lam=lam)

x0 = torch.ones((1, d), dtype=dtype, device=device)

T = 100000

b = 1

h = lambda k : 1e-5
reps = 10

#gammas = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2] #  
gammas = [0.001, 0.01, 0.1, 1.0]
f0 = target.full_target(x0, elem_wise=False).item()
for (i, gamma) in enumerate(gammas):
    print(f"[{opt_name}] gamma = {gamma}")
    exp_name = f"{opt_name}_{l}_{m}_{b}"
    cost_per_iter = get_cost_per_iter(opt_name, d = d, l = l, m = m, n = d, b = b)
    num_iters = T // cost_per_iter
    opt = get_optimizer(name=opt_name, d = d, l = l, prox = prox, b = b, dtype = dtype, device = device, seed = seed)
    opt_result = test_optimizer(target=target, optimizer=opt, x0=x0, T = num_iters, m = m, gamma=gamma, h= h, reps=reps)

    mu_vals, std_vals = opt_result['values']
    mu_time, std_time = opt_result['times']

    if mu_vals[-1] != mu_vals[-1] or mu_vals[-1] + std_vals[-1] > f0:
        with open(f"{out_dir}/param_tuning/{exp_name}.log", 'a') as f:
            for j in range(len(gammas[i:])):
                f.write(f"{f0},{f0},{0.0},{mu_time[-1]},{std_time[-1]},{gammas[i + j]}\n")
        break
    else:
        with open(f"{out_dir}/param_tuning/{exp_name}.log", 'a') as f:
            f.write(f"{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]},{gamma}\n")
        with open(f"{out_dir}/full_results/{opt_name}_{l}_{m}_{b}_{gamma}.log", 'a') as f:
            for i in range(len(mu_vals)):
                f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost_per_iter}\n")     


