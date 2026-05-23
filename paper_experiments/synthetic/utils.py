import sys
import torch
import numpy as np 
from svrz.optimizers import VRSZD, VRSZDPhase

from svrz.prox import SoftThreshold, ProxOperator


sys.path.append("../")
from targets import QuadraticLasso




class SyntheticResult:

    def __init__(self):
        self.full_values = []
        self.grad_map_norms = []
        self.list_num_evals = []
    
    def add(self, values, gmaps, num_evals):
        self.full_values.append(values)
        self.grad_map_norms.append(gmaps)
        self.list_num_evals.append(num_evals)

    def get_stats(self):
        mu_values, std_values         = np.mean(self.full_values, axis=0),    np.std(self.full_values, axis=0)
        mu_gmap_norms, std_gmap_norms = np.mean(self.grad_map_norms, axis=0), np.std(self.grad_map_norms, axis=0)
        mu_num_evals, std_num_evals   = np.mean(self.list_num_evals, axis=0), np.std(self.list_num_evals, axis=0)
        return mu_values, std_values, mu_gmap_norms, std_gmap_norms, mu_num_evals, std_num_evals

    


def get_target_function(name, d, target_params, seed = 1213, dtype=torch.float64, device='cuda'):
    if name == 'qlasso':
        L, mu, lam = target_params['L'], target_params['mu'], target_params['lam']
        prox = SoftThreshold(lam=lam)
        return QuadraticLasso(d = d,  L = L, mu = mu, lam=lam, dtype=dtype, device=device, seed = seed), prox




def get_gmap_norm(target, x, gamma, prox):
    grad_x = target.grad(x)
    grad_map_x = (x - prox(x - gamma * grad_x, gamma)) / gamma
    return grad_map_x.norm(p=2).square().item()



def run_vrszd(opt, target, budget):
    num_evals = 0
    f_map = torch.vmap(torch.vmap(lambda x, z: target(x, z), in_dims=(0, None)), in_dims=(0,0))
    full_values = [target.full_target(opt.x0).item()]
    grad_map_norms = [get_gmap_norm(target, opt.x0, opt.gamma(opt.tau), opt.prox)]

    list_num_evals = [0]
    initialization = True
    batch_size = opt.P.b
    d, l = target.d, opt.P.l
    while num_evals < budget:
        population = opt.ask()
        if opt.phase == VRSZDPhase.DET_GRAD_APPROX:
            x_next, x0 = population
            f_next = torch.vmap(lambda x: target(x), in_dims=(0,))(x_next)
            f_curr = target(x0)
            values = [f_next, f_curr]

            num_evals += target.n * (d + 1)
            
            if not initialization:
                full_value = target.full_target(x0)
                full_values.append(full_value.item())
                grad_map_norms.append(get_gmap_norm(target, x0, opt.gamma(opt.tau), opt.prox))
                list_num_evals.append(num_evals)
            initialization = False
            print(f"[{num_evals}/{budget}] F(x_tau) = {full_values[-1]/full_values[0]}\t||G(x)||^2 = {grad_map_norms[-1]/grad_map_norms[0]}")
        else:
            x_inn_next, x_out_next, x_inner, x_out = population

            zeta = target.sample_z(batch_size) 

            f_inn_next, f_out_next = f_map(x_inn_next, zeta), f_map(x_out_next, zeta)

            f_inn_curr  = torch.vmap(lambda z : target(x_inner, z), in_dims=(0,))(zeta)
            f_out_curr  = torch.vmap(lambda z : target(x_out, z), in_dims=(0, ))(zeta)

            values = [f_inn_next, f_out_next, f_inn_curr, f_out_curr]
            num_evals += np.sum([x.flatten().shape[0] for x in values])

        opt.tell(population, values)
    return full_values, grad_map_norms, list_num_evals