import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, Dict
from .regularizer import RegularizerAbstract
from .dataset import SPOnlineDataset

def train(epochs: int,
          dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          data_loss_fns: Dict[str, torch.nn.Module],
          data_loss_weights: Dict[str, float],
          bayesian_loss_fn: torch.nn.Module,
          bayesian_loss_weight: float,
          regularizer: Tuple[RegularizerAbstract],
          device: torch.device) -> None:
    """
    Train the model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function to compute the loss.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    model.train()
    model.to(device)
    with tqdm(total=epochs, unit="epoch") as pbar:
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                inputs = regularizer[0].regularize(inputs)
                for key in targets.keys():
                    targets[key] = targets[key].to(device)
                    targets[key] = regularizer[1][key].regularize(targets[key])
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = bayesian_loss_weight * bayesian_loss_fn(model)
                for key in model.names:
                    loss += data_loss_weights[key] * data_loss_fns[key](outputs[key], targets[key])
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(dataloader)
            if epoch % 10 == 9:
                # update the progress bar
                pbar.set_postfix_str(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')
                pbar.update(10)
    return model

def validate(inputs,
             outputs,
             model: torch.nn.Module,
             regularizer: Tuple[RegularizerAbstract],
             samples: int,
             device: torch.device):
    """
    Validate the model on the validation dataset.

    Args:
    - inputs (torch.Tensor): Input data for validation.
    - outputs (torch.Tensor): Target data for validation.
    - model (torch.nn.Module): The model to validate.
    - regularizer (Tuple[RegularizerAbstract]): Regularizer to apply to inputs and outputs.
    - samples (int): Number of samples to generate for uncertainty estimation.
    - device (torch.device): Device to run the validation on (CPU or GPU).

    Returns:
        Dict[str, float]: Dictionary containing average loss for each output.
    """    
    inputs = inputs.to(device)
    inputs = regularizer[0].regularize(inputs)
            
    output_samples = {key: [] for key in model.names}
    for s in range(samples):
        outputs = model(inputs)
        for key in outputs.keys():
            outputs[key] = regularizer[1][key].deregularize(outputs[key])
            output_samples[key].append(outputs[key].cpu().detach().numpy())
            
    # aggregate the outputs
    for key in output_samples.keys():
        output_samples[key] = np.stack(output_samples[key], axis=1)

    return output_samples