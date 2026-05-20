import os
import sys
import torch

import argparse as ap 

from svrz.prox import SoftThreshold
from svrz.optimizers import VRSZD
from svrz.directions import QRDirections, SphericalDirections

sys.path.append("../")


from targets import LeastSquares
from datasets import SyntheticDataset
from utils import get_optimizer, get_cost_per_iter, test_optimizer




L = float(sys.argv[1]) #if len(sys.argv) > 2 else 10.0
mu = float(sys.argv[2]) #if len(sys.argv) > 3 else 1.0

d = int(sys.argv[3]) 
m = int(sys.argv[4]) 
l = int(sys.argv[5]) 

base_out_dir = sys.argv[6]

out_dir = f"{base_out_dir}/struct_vs_unstruct/lasso_{L}_{mu}_{d}"

os.makedirs(out_dir, exist_ok=True)

dtype = torch.float64
device = 'cuda'
x_start = torch.zeros((d, ), dtype=dtype, device=device)

seed = 12131415
lam = 1e-5

data = SyntheticDataset(x_star=x_start, n = d, L = L, mu = mu, seed=seed)

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
        with open(f"{out_dir}/full_results/_{l}_{m}_{b}_{gamma}.log", 'a') as f:
            for i in range(len(mu_vals)):
                f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost_per_iter}\n")     

def main(args):
    direction_type = args.direction_type
    L, mu = args.L, args.mu
    d = args.d
    gamma = args.gamma
    m, l, b = args.m, args.l, args.b
    base_out_dir = args.out_dir
    base_seed = args.seed

    dir_seed = 432* base_seed + 123

    if direction_type == 'structured':
        direction_sampler = QRDirections(d = d, l = l, seed = dir_seed)
    else:
        direction_sampler = SphericalDirections(d = d, l = l, seed = dir_seed)



if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Ablation on structure vs unstructured", formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument('direction_type', type=str, choices=['structured', 'unstructured'], help='Direction type')
    parser.add_argument('--L', type=float, default = 1e5, help='L')
    parser.add_argument('--mu', type=float, default=1.0, help='mu')
    parser.add_argument('--d', type=int, default=10, help='dimension of input space')
    parser.add_argument('--gamma', type=float, default=1e-3, help='gamma')
    parser.add_argument('--m', type=int, help='number of inner loop iteration')
    parser.add_argument('--l', type=int, help='number of directions')
    parser.add_argument('--b', type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=115, help='random seed')
    parser.add_argument('--out-dir', type=str, help='output directory')
    args = parser.parse_args()
 #   main(args)