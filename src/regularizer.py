from abc import ABC, abstractmethod
import torch
class RegularizerAbstract(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def regularize(self, n):
        pass

    @abstractmethod
    def deregularize(self, n):
        pass

class NoneRegularizer(RegularizerAbstract):
    def __init__(self):
        super().__init__()
    
    def regularize(self, n):
        return n

    def deregularize(self, n):
        return n
    
class MinMaxRegularizer(RegularizerAbstract):
    def __init__(self, min_n, max_n) -> None:
        super().__init__()
        self.min_n = torch.tensor(min_n, dtype=torch.float32)
        self.max_n = torch.tensor(max_n, dtype=torch.float32)
    
    def regularize(self, n):
        """
        Regularize the input n to be within the range [min_n, max_n].
        
        Args:
            n (torch.Tensor): Input tensor to be regularized.
        
        Returns:
            torch.Tensor: Regularized tensor.
        """
        self.min_n = self.min_n.to(n.device)
        self.max_n = self.max_n.to(n.device)
        return (n - self.min_n) / (self.max_n - self.min_n) * 2 - 1
    
    def deregularize(self, n):
        """
        Deregularize the input n back to its original scale.
        
        Args:
            n (torch.Tensor): Input tensor to be deregularized.
        
        Returns:
            torch.Tensor: Deregularized tensor.
        """
        self.min_n = self.min_n.to(n.device)
        self.max_n = self.max_n.to(n.device)
        return (n + 1) / 2 * (self.max_n - self.min_n) + self.min_n

class DivideRegularizer(RegularizerAbstract):
    def __init__(self, divider) -> None:
        super().__init__()
        self.divider = torch.tensor(divider, dtype=torch.float32)
    
    def regularize(self, n):
        """
        Regularize the input n using standardization.
        
        Args:
            n (torch.Tensor): Input tensor to be regularized.
        
        Returns:
            torch.Tensor: Regularized tensor.
        """
        return n / self.divider.to(n.device)
    
    def deregularize(self, n):
        """
        Deregularize the input n back to its original scale.
        
        Args:
            n (torch.Tensor): Input tensor to be deregularized.
        
        Returns:
            torch.Tensor: Deregularized tensor.
        """
        return n * self.divider.to(n.device)