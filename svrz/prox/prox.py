import torch

class ProxOperator:
        
    def __call__(self, x, gamma):
        return x
    
        
class SoftThreshold(ProxOperator):
    
    def __init__(self, lam) -> None:
        self.lam = lam
        
    def __call__(self, x, gamma):
        return torch.sign(x) * torch.maximum(abs(x) - self.lam * gamma, torch.zeros_like(x))
    

