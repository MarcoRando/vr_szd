
from torch import Tensor, Generator, dtype, float32
from typing import Dict, Optional, Callable
from svrz.directions import DirectionGenerator
from svrz.utils import TargetFunction
from svrz.prox import ProxOperator

class AbsOptimizer:
    
  def __init__(self, 
              x0 : Tensor,
              P : DirectionGenerator, 
              P_full : Optional[DirectionGenerator] = None, 
              prox : ProxOperator | None = None,                 
              seed : int = 121314, 
              device : str = 'cpu',
              dtype : dtype = float32) -> None:
      self.x0 = x0
      self.P = P
      self.P_full = P_full if P_full is not None else P
      self.dtype = dtype
      self.device = device
      self.prox = prox if prox is not None else ProxOperator()
      self.generator = Generator(device=device)
      self.generator.manual_seed(seed)
    
  def ask(self):
    raise NotImplementedError("The ask method is implemented in subclasses!")

  def tell(self, X, y):
    raise NotImplementedError("The tell method is implemented in subclasses!")
            
  def optimize(self, 
                f : TargetFunction, # objective function
                x0 : Tensor,  # initial guess
                T : int,  # number of iterations
                gamma : Callable[[int], float],  # stepsize
                h : Callable[[int], float], # discretization parameter
                callback : Callable[[Tensor, float, int], None] | None = None # callback
              ) -> Dict:
    raise NotImplementedError("Optimize method is implemented in subclasses!")