import os
import pickle
import re
from typing import List, Sequence, Tuple
import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

from .poles_init import basic_poles_initialize
from .utils import float_p 
from .touchstone import TouchstoneParser
from .complex_func import FunctionFromTouchstone
from .vector_fitting import vector_fit
from .loader import load_dataset

# macro to control whether to regenerate the data and store them
PKL_AGAIN = False
PR_AGAIN = False

def load_poles_residues_dataset(dataset_path, port_num: int, function_from_touchstone: FunctionFromTouchstone, freqs: np.ndarray, init_poles, force_regen: bool = False, force_regen_pr: bool = False, random_sample: bool = False, port_combinations: List = None):
    dataset = load_dataset(dataset_path, port_num, function_from_touchstone, force_regen, random_sample)
    size = len(dataset)
    npc = len(port_combinations)
    npoles = init_poles.shape[0]
    pr_dataset = []

    for i in range(size):
        # the case of poles and residues are related to the snp file
        filename = f"poles_residues_{i}.pkl"
        file_path = os.path.join(dataset_path, filename)
        
        if os.path.exists(file_path) and not force_regen_pr:
            with open(file_path, 'rb') as f:
                pr_data = pickle.load(f)
                pr_dataset.append(pr_data)
                continue

        # initiate the s params for vector fitting
        data = dataset[i]
        s_params = data["s_params"].reshape(port_num, port_num, -1)
        
        poles_array, residues_array = np.zeros((npc, npoles), dtype=complex),  np.zeros((npc, npoles), dtype=complex)
        d_array, h_array = np.zeros((npc, 1)), np.zeros((npc, 1))
        
        # vector fit all the combinations
        for pci, pc in enumerate(port_combinations):
            poles, residues, d, h = vector_fit(freqs, s_params[pc[0], pc[1]], init_poles, tol=1e-8, atol=1e-8, rtol=1e-8)
            poles_array[pci] = poles
            residues_array[pci] = residues
            d_array[pci] = d
            h_array[pci] = h

        #### Temp assumption npc == 1
        poles_array = poles_array.reshape(-1)
        residues_array = residues_array.reshape(-1)
        d_array = d_array.reshape(-1)
        h_array = h_array.reshape(-1)


        pr_data = {
            "caseid": i,
            "geo_params": data["geo_params"],
            "s_params": data["s_params"],
            "sampled_indices": data["sampled_indices"],
            "poles": poles_array,
            "residues": residues_array,
            "d": d_array,
            "h": h_array,
        }
        pr_dataset.append(pr_data)

        with open(file_path, "wb") as f:
            pickle.dump(pr_data, f)

    return pr_dataset
        
def save_data(data, dataset_path):
    """
    Save the dataset to a specified path.
    
    Args:
    - dataset: The dataset to be saved.
    - dataset_path: Path to save the dataset.
    """
    # check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    caseid = data["caseid"]
    # geo_params = data["geo_params"]
    # s_params = data["s_params"]
    filename = f"case_{caseid}.pkl"
    file_path = os.path.join(dataset_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

class SPDataset(Dataset):
    """
    A PyTorch Dataset class for loading S-parameter data from a given dataset.
    
    Args:
    - dataset_path (str): Path to the dataset directory.
    - port_num (int): Number of ports in the S-parameter files.
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 freqs: np.ndarray, 
                 port_num: int, 
                 param_start_idx: int,
                 port_combinations: List[Tuple[int]] = None, 
                 mode: str = "ri", 
                 random_sample: bool = False,):
        
        self.dataset_path = dataset_path
        self.port_num = port_num
        self.param_start_idx = param_start_idx
        self.port_combinations = port_combinations
        self.function_from_touchstone = FunctionFromTouchstone(port_num, None)
        self.freqs = torch.tensor(freqs, dtype=torch.float32).reshape(-1, 1)
        self.random_sample = random_sample
        self.dataset = load_dataset(dataset_path, port_num, self.function_from_touchstone, self.param_start_idx, force_regen=PKL_AGAIN, random_sample=self.random_sample)
        self.mode = mode

    def flatten_geo_params(self, geo_params: torch.Tensor, sampled_indices: List[int] = None):
        """
        Flatten the geometric parameters and concatenate them with frequency points.
        Args:
        - geo_params (torch.Tensor): Geometric parameters.
        - sampled_indices (List[int]): Indices to sample from the geometric parameters.
        Returns:
        - input_features (torch.Tensor): Resembled input features containing geometric parameters and frequency points.
        """
        
        freq_num = self.freqs.shape[0]
        geo_params = geo_params.repeat((freq_num, 1))
        # concatenate frequency data with geometric parameters
        input_features = torch.cat((geo_params, self.freqs), dim=1)
        if sampled_indices is not None and self.random_sample:
            # sample the indices if provided
            input_features = input_features[sampled_indices]
        return input_features
    
    def flatten_s_parameters(self, s_params: torch.Tensor, sampled_indices: List[int] = None, flatten: bool = True):
        """
        Flatten the frequency data for S-parameters.
        Args:
        - s_params (torch.Tensor): S-parameter data.
        - sampled_indices (List[int]): Indices to sample from the geometric parameters.
        - flatten (bool): Whether to flatten the S-parameters.
        Returns:
        - s_params (torch.Tensor): Reshaped S-parameter data.
        """
        
        if self.port_combinations is not None:
            # reshape S-parameters for specific port combinations
            select = []
            for pc in self.port_combinations:
                select.append(s_params[pc[0], pc[1], :])
            s_params = torch.stack(select, dim=0) # shape (ports, freqs)

        if flatten:
            if sampled_indices is not None and self.random_sample:
                # sample the indices if provided
                s_params = s_params[:, sampled_indices]
            
            s_params = s_params.permute(1, 0) # shape (freqs, ports)

        return s_params
    
    def load(self, dataset_path: str):
        """
        Load the dataset from a specified path.
        
        Args:
        - dataset_path (str): Path to load the dataset from.
        """
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
            
    def random_sample_frequency(self, sample_num: int):
        assert sample_num <= len(self.freqs), "Sample number exceeds available frequency points."
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
        - idx (int): Index of the item to retrieve.
        
        Returns:
        - A dictionary containing the case ID, parameters, and S-parameter data.
        """
        data = self.dataset[idx]
        geo_params = torch.tensor(data["geo_params"], dtype=torch.float32)
        if self.mode == "abs":
            s_params = torch.tensor(np.abs(data["s_params"]), dtype=torch.float32).unsqueeze(0)
            input_features, s_params = self.flatten_frequency(geo_params, s_params, data["sampled_indices"])
            return input_features, s_params
        elif self.mode == "ri":
            input_features = self.flatten_geo_params(torch.tensor(data["geo_params"], dtype=torch.float32), data["sampled_indices"])
            s_params_real = self.flatten_s_parameters(torch.tensor(np.real(data["s_params"]), dtype=torch.float32), data["sampled_indices"])
            s_params_imag = self.flatten_s_parameters(torch.tensor(np.imag(data["s_params"]), dtype=torch.float32), data["sampled_indices"])
            return input_features, {"r": s_params_real, "i": s_params_imag}
        elif self.mode == "unflatten_ri":
            input_features = torch.tensor(data["geo_params"], dtype=torch.float32)
            s_params_real = self.flatten_s_parameters(torch.tensor(np.real(data["s_params"]), dtype=torch.float32), data["sampled_indices"], False)
            s_params_imag = self.flatten_s_parameters(torch.tensor(np.imag(data["s_params"]), dtype=torch.float32), data["sampled_indices"], False)
            return input_features, {"r": s_params_real, "i": s_params_imag}
        else:
            raise ValueError()
        

class PolesResiduesDataset(Dataset):
    """
    A PyTorch Dataset class for loading pole residues data from a given dataset.
    
    Args:
    - dataset_path (str): Path to the dataset directory.
    - port_num (int): Number of ports in the S-parameter files.
    """
    
    def __init__(self, dataset_path, freqs, port_num, init_poles, n_poles_real, n_poles_complex, port_combinations = [(0, 0)], random_sample: bool = False):
        self.dataset_path = dataset_path
        self.port_num = port_num
        self.port_combinations = port_combinations
        self.freqs = freqs
        self.init_poles = init_poles
        self.n_poles_real = n_poles_real
        self.n_poles_complex = n_poles_complex
        self.function_from_touchstone = FunctionFromTouchstone(port_num, None)
        self.dataset = load_poles_residues_dataset(dataset_path, port_num, self.function_from_touchstone, self.freqs, init_poles, force_regen=PKL_AGAIN, force_regen_pr=PR_AGAIN, random_sample=random_sample, port_combinations=port_combinations)
        self.mode = None

    def post_process(self, items):
        """
        Realize the poles and residues.
        
        Args:
        - items (torch.Tensor): Poles or residues values.
        
        Returns:
        - Realized poles and residues.
        """
        # real_part = torch.real(items[:, :self.n_poles_real])
        # complex_part_real = torch.real(items[:, self.n_poles_real:self.n_poles_real+self.n_poles_complex*2:2])
        # complex_part_imag = torch.imag(items[:, self.n_poles_real:self.n_poles_real+self.n_poles_complex*2:2])

        real_part = torch.real(items[:self.n_poles_real])
        complex_part_real = torch.real(items[self.n_poles_real:self.n_poles_real+self.n_poles_complex*2:2])
        complex_part_imag = torch.imag(items[self.n_poles_real:self.n_poles_real+self.n_poles_complex*2:2])

        return torch.cat((real_part, complex_part_real, complex_part_imag), dim=0)

    def set_output_features(self, mode: str):
        self.mode = mode

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        geo_params = torch.tensor(data["geo_params"], dtype=torch.float32)
        poles = self.post_process(torch.tensor(data["poles"], dtype=torch.complex64))
        residues = self.post_process(torch.tensor(data["residues"], dtype=torch.complex64))
        d = torch.tensor(data["d"], dtype=torch.float32)
        h = torch.tensor(data["h"], dtype=torch.float32)
        return geo_params, poles, residues, d, h

class SPOnlineDataset(SPDataset):

    """
    A PyTorch Dataset class for loading S-parameter data from a given dataset.
    
    Args:
    - dataset_path (str): Path to the dataset directory.
    - port_num (int): Number of ports in the S-parameter files.
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 freqs: np.ndarray, 
                 port_num: int, 
                 param_start_idx: int,
                 port_combinations: List[Tuple[int]] = None, 
                 mode: str = "ri", 
                 random_sample: bool = False,):
        super().__init__(dataset_path, freqs, port_num, param_start_idx, port_combinations, mode, random_sample)
        self.available_idx = np.arange(len(self.dataset)).tolist() # available indices in the dataset at first
        self.sample_idx = []
        self.history_idx = []

    def random_sample_idx(self, sample_size: int):
        """
        Randomly sample the indices from the dataset.
        
        Args:
        - sample_size (int): Size of the sanpling set.
        """
        if sample_size > len(self.dataset):
            print("Train size exceeds dataset size.")
            self.sample_idx = self.available_idx
        else:
            self.sample_idx = np.random.choice(self.available_idx, sample_size, replace=False).tolist()

    def remove_idx(self, indices):
        """
        Remove a specific index from the avaiable indices.
        
        Args:
        - indices (int): Index to be removed.
        """
        # do nothing
        for idx in indices:
            for i, idx_ in enumerate(self.available_idx):
                if idx_ == idx:
                    del self.available_idx[i]
                    break
        return
    
    def set_sample_idx(self, sample_idx: List[int]):
        """
        Set the sample indices for the dataset.
        
        Args:
        - sample_idx (List[int]): List of indices to set as sample indices.
        """
        self.sample_idx = sample_idx

    def set_available_idx(self, available_idx: List[int]):
        """
        Set the available indices for the dataset.
        
        Args:
        - available_idx (List[int]): List of indices to set as available indices.
        """
        self.available_idx = available_idx

    def extend_sample_idx(self, sample_idx: List[int]):
        """
        Append new sample indices to the existing sample indices.
        
        Args:
        - sample_idx (List[int]): List of indices to append.
        """
        self.sample_idx.extend(sample_idx)

    def extend_history_idx(self, history_idx: List[int]):
        """
        Append new history indices to the existing history indices.
        
        Args:
        - history_idx (List[int]): List of indices to append.
        """
        self.history_idx.extend(history_idx)

    def get_sample_idx(self):
        """
        Get the current sample indices.
        
        Returns:
        - List of sample indices.
        """
        return self.sample_idx
    
    def get_available_idx(self):
        """
        Get the current available indices.
        
        Returns:
        - List of available indices.
        """
        return self.available_idx

    def reset_available_idx(self):
        """
        Reset the available indices to the original dataset size.
        """
        self.available_idx = np.arange(len(self.dataset)).tolist()

    def save(self, dataset_path: str):
        """
        Save the current dataset to a specified path.
        
        Args:
        - dataset_path (str): Path to save the dataset.
        """
        data = {
            "sample_idx": self.sample_idx,
            "available_idx": self.available_idx,
            "history_idx": self.history_idx,
            "dataset": self.dataset,
        }
        with open(dataset_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, dataset_path: str):
        """
        Load the dataset from a specified path.
        
        Args:
        - dataset_path (str): Path to load the dataset from.
        """
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            self.sample_idx = data["sample_idx"]
            self.available_idx = data["available_idx"]
            self.history_idx = data["history_idx"]
            self.dataset = data["dataset"]

    def __len__(self):
        return len(self.sample_idx)
    
    def __getitem__(self, idx):
        return super().__getitem__(self.sample_idx[idx])
    


def test_SPDataset():
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    freqs = np.arange(0.1e9, 20e9, 0.03e9)
    port_combinations = [(0, 0), (0, 1)]
    dataset = SPDataset(dataset_path, freqs, port_num, 4, port_combinations, mode="unflatten_ri")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    for x, y in dataloader:
        geo_params = x
        s_params = y
        print(f"Geo Params: {geo_params.shape}, S-Params: {s_params['r'].shape}")

def test_PolesResiduesDataset():
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    freqs = np.arange(0.1e9, 20e9, 0.03e9)
    init_poles = basic_poles_initialize(freqs_range=(0.1e9, 20e9), n_poles_real=4, n_poles_complex=2)
    port_combinations = [(0, 0)]
    dataset = PolesResiduesDataset(dataset_path, freqs, port_num, init_poles, 6, 1, port_combinations, random_sample=True)
    for i in range(10):

        dataset.set_output_features("p")  # Set mode to 'p' for poles
        geo_params, poles = dataset[i]
        print("geo_params: ", geo_params)
        print("poles: ", poles)
        dataset.set_output_features("r")
        _, residues = dataset[i]
        print("residues: ", residues)
        dataset.set_output_features("d")
        _, d = dataset[i]
        print("d: ", d)
        dataset.set_output_features("h")
        _, h = dataset[i]
        print("h: ", h)

def test_SPOnlineDataset():
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    freqs = np.arange(0.1e9, 20e9, 0.03e9)
    port_combinations = [(0, 0), (0, 1)]
    dataset = SPOnlineDataset(dataset_path, freqs, port_num, 4, port_combinations)
    
    # Initialize sample indices
    print("Available indices after initialization:", len(dataset.available_idx))

    dataset.random_sample_idx(10)
    dataset.remove_idx(dataset.sample_idx)
    print("Available indices after initialization:", len(dataset.available_idx))
    return
    for i in range(len(dataset)):
        geo_params, s_params = dataset[i]
        print(f"Case ID: {dataset.sample_idx[i]}, Geo Params: {geo_params}, S-Params: {s_params}")