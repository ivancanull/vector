import os
import pickle
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import float_p 
from .touchstone import TouchstoneParser
from .complex_func import FunctionFromTouchstone

PKL_AGAIN = False

def load_dataset(dataset_path, port_num: int, function_from_touchstone: FunctionFromTouchstone = None, force_regen: bool = False, random_sample: bool = False):
    """
    Load the dataset containing S-parameters snp files.
    
    Args:
    - dataset_path: Path to the dataset directory.
    - port_num: Number of ports in the S-parameter files.

    Returns:
    - dataset: A list containing the S-parameter data.
    """
    # Check if the dataset path exists
    
    dataset = []
    for filename in tqdm(os.listdir(dataset_path), desc="Loading dataset", unit="file"):
        if filename.endswith(f".s{port_num}p"):
            file_path = os.path.join(dataset_path, filename) 
            
            # file name is {case_name}_{HFSSDesign}{id}_{params}.snp
            # find 'HFSSDesign' and split by '_'

            splits = filename.split('.')[0].split('_')
            while splits[0].find('HFSSDesign') == -1:
                splits = splits[1:]
            caseid = splits[0][10:] # HFSSDesign{caseid} remove 'HFSSDesign'
            splits = splits[1:]
            # assert caseid is a number
            assert int(caseid) >= 0, f"Invalid caseid: {caseid}"
            caseid = int(caseid)

            # check if the pkl file exists
            if os.path.exists(os.path.join(dataset_path, f"case_{caseid}.pkl")) and not force_regen:
                with open(os.path.join(dataset_path, f"case_{caseid}.pkl"), 'rb') as f:
                    single_case = pickle.load(f)
                    dataset.append({
                        "caseid": caseid,
                        "geo_params": single_case["geo_params"],
                        "s_params": single_case["s_params"],
                        "sampled_indices": single_case["sampled_indices"] # add sampled indices for training
                    })
                    continue

            geo_param_vals = []
            while splits:
                assert float_p(splits[1]) >= 0, f"Invalid param: {splits[1]}"
                geo_param_vals.append(float_p(splits[1]))
                splits = splits[2:]


            # load the touchstone file
            function_from_touchstone.parse(file_path)
            # print(f"Function from touchstone: {function_from_touchstone.data}")
            s_params = function_from_touchstone.evaluate_all()

            # generate sampled indices if random_sample is True
            if random_sample:
                sampled_indices = np.random.choice(len(function_from_touchstone.freq), size=10, replace=False)
            else:
                sampled_indices = None
            
            single_case = {
                "caseid": caseid,
                "geo_params": geo_param_vals,
                "s_params": s_params,
                "sampled_indices": sampled_indices
            }
            save_data(single_case, dataset_path) # save the dataset for future easier access
            dataset.append(single_case)

    return dataset

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
    
    def __init__(self, dataset_path, freqs, port_num, port_combinations = None, mode: str = "abs", random_sample: bool = False):
        self.dataset_path = dataset_path
        self.port_num = port_num
        self.port_combinations = port_combinations
        self.function_from_touchstone = FunctionFromTouchstone(port_num, None)
        self.freqs = torch.tensor(freqs, dtype=torch.float32).reshape(-1, 1)
        self.random_sample = random_sample
        self.dataset = load_dataset(dataset_path, port_num, self.function_from_touchstone, force_regen=PKL_AGAIN, random_sample=self.random_sample)
        self.mode = mode

    def flatten_frequency(self, geo_params: torch.Tensor, s_params: torch.Tensor, sampled_indices: List[int] = None):
        """
        Flatten the frequency data for S-parameters.
        
        Args:
        - geo_params (torch.Tensor): Geometric parameters.
        - s_params (torch.Tensor): S-parameter data.
        
        Returns:
        - resembled input features (torch.Tensor) containing geometric parameters and frequency points.
        - s_params (torch.Tensor): Reshaped S-parameter data.

        """
        freq_num = self.freqs.shape[0]
        geo_params = geo_params.repeat((freq_num, 1))
        if self.port_combinations is None:
            # Reshape S-parameters for all port combinations
            s_params = s_params.reshape((self.port_num, self.port_num, freq_num))
            s_params = s_params.permute(2, 0, 1).reshape(freq_num, -1)
        else:
            # Reshape S-parameters for specific port combinations
            i, j = self.port_combinations
            s_params = s_params[i, j].reshape(freq_num, 1)

        # Concatenate frequency data with geometric parameters
        input_features = torch.cat((geo_params, self.freqs), dim=1)
        if sampled_indices is not None and self.random_sample:
            # Sample the indices if provided
            input_features = input_features[sampled_indices]
            s_params = s_params[sampled_indices]
        
        return input_features, s_params

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
            s_params = torch.tensor(np.abs(data["s_params"]), dtype=torch.float32)
        
        input_features, s_params = self.flatten_frequency(geo_params, s_params, data["sampled_indices"])
        return input_features, s_params

def test_SPDataset():
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    dataset = SPDataset(dataset_path, port_num)
    
    for i in range(len(dataset)):
        geo_params, s_params = dataset[i]
        # print(f"Case ID: {i}, Geo Params: {geo_params}, S-Params: {s_params}")