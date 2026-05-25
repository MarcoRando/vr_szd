import os
import sys
import torch

import numpy as np
import argparse as ap
import fcntl

from svrz.optimizers import VRSZD, VRSZDPhase
from svrz.prox import SoftThreshold, ProxOperator
from svrz.directions import QRDirections, SphericalDirections


from utils import get_target_function, run_vrszd, SyntheticResult, get_gmap_norm



def get_algorithm(name, params, x0, prox, seed = 1213, dtype=torch.float64, device='cuda'):
    d, gamma, h = params[0], params[1], params[2]
    if name == 'vrszd':
        l, b, m, directions = params[3], params[4], params[5], params[6]

        P = QRDirections(d = d, l=l, b=b, seed = seed, dtype=dtype, device=device) if directions == 'structured' else SphericalDirections(d = d, l=l, b=b, seed = seed, dtype=dtype, device=device)

        opt = VRSZD(x0 = x0, P = P, m = m, gamma=gamma, h=h, prox = prox, seed = seed)
        return opt, run_vrszd


def run_experiment(algo_name, params, target, prox, reps, budget, dtype, device, base_seed):

    initialization_seed = 432*base_seed + 133
    directions_seed = 123*base_seed + 135

    init_generator = torch.Generator(device=device).manual_seed(initialization_seed)

    results = SyntheticResult()
    for i in range(reps):
        init_eps = torch.rand((target.d,), dtype=dtype, device=device, generator=init_generator)
        x0 = (target.bounds[:, 1] - target.bounds[:, 0]) * init_eps + target.bounds[:, 0]

        opt, exp_runner = get_algorithm(algo_name, params, x0, prox, seed = directions_seed, dtype=dtype, device=device)
        full_values, grad_map_norms, list_num_evals= exp_runner(opt, target, budget)
        full_values = [ fval / full_values[0] for fval in full_values]
        grad_map_norms = [gm_norm / grad_map_norms[0] for gm_norm in grad_map_norms]
        results.add(full_values, grad_map_norms, list_num_evals)
    return results


def get_algo_params(algo_name, args):
    algo_params = [args.d, args.gamma, args.h] #dict(d=args.d, gamma=args.gamma, h=args.h, seed=args.seed)
    if algo_name == 'vrszd':
        algo_params += [args.l, args.b, args.m, args.directions]

    return algo_params


def main(args):

    algo_name = args.algorithm
    target_name =  args.target

    d, budget = args.d, args.budget


    reps, base_seed = args.reps, args.seed
    
    device = args.device
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    base_out_dir = args.out_dir

    out_dir = f"{base_out_dir}/synthetic"
    os.makedirs(out_dir, exist_ok=True)


    target_params = dict(d=d, lam=args.lam)
    if target_name == 'qlasso':
        target_params['L'] = args.L
        target_params['mu'] = args.mu

    algo_params = get_algo_params(algo_name, args)


    target, prox = get_target_function(target_name, d, target_params, seed = base_seed, dtype=dtype, device=device)

    results = run_experiment(algo_name, algo_params, target, prox, reps, budget, dtype, device, base_seed)

    mu_fevals, std_fevals, mu_gmap_norms, std_gmap_norms, mu_num_evals, std_num_evals = results.get_stats()
    with open(f"{out_dir}/{algo_name}_{target_name}_{d}_results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(",".join([str(x) for x in algo_params]) + f",{mu_fevals[-1]},{std_fevals[-1]},{mu_gmap_norms[-1]},{std_gmap_norms[-1]}\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)




if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Run synthetic experiment", formatter_class = ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument("algorithm", type=str, default='vrszd', choices=['vrszd'], help="Name of the algorithm")
    parser.add_argument("target", type=str, default='qlasso', choices=['qlasso'], help="Name of the target function")

    # Experiment parameters
    parser.add_argument("--d", type=int, default=5, help="Dimension of the problem")
    parser.add_argument("--L", type=float, default=100000.0, help="Lipschitz constant of the gradient of the target function f")
    parser.add_argument("--mu", type=float, default=1.0, help="Strong convexity parameter of the target function f")
    parser.add_argument("--lam", type=float, default=1e-3, help="Regularization parameter of the regularization term h")
    parser.add_argument("--budget", type=int, default=1000000, help='Number of (stochastic) function evaluations')

    # VR-SZD parameters
    parser.add_argument("--b", type=int, default=3, help="Batch size (i.e. number of rotations G)")
    parser.add_argument("--l", type=int, default=5, help="Number of directions")
    parser.add_argument("--m", type=int, default=10, help="Number of inner loop iterations")
    parser.add_argument("--gamma", type=float, default=1e-5, help="Stepsize")
    parser.add_argument("--h", type=float, default=1e-5, help="Smoothing parameter")
    parser.add_argument("--directions", type=str, default='structured', choices=['structured', 'unstructured'], help="Type of directions to use")

    # General parameters
    parser.add_argument("--reps", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--seed", type=int, default=1213, help="Base random seed")
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda'], help="Device to use")
    parser.add_argument("--dtype", type=str, default='float64', choices=['float32', 'float64'], help="Data type to use")
    parser.add_argument("--out_dir", type=str, default='./', help="Output directory")


    args = parser.parse_args()
    main(args)