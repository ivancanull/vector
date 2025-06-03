import os
import pickle
import numpy as np
from tqdm import tqdm
from .complex_func import FunctionFromTouchstone
from .utils import float_p 

def load_dataset(dataset_path, port_num: int, function_from_touchstone: FunctionFromTouchstone, param_start_idx: int, force_regen: bool = False, random_sample: bool = False):
    """
    Load the dataset containing S-parameters snp files.
    
    Args:
    - dataset_path: Path to the dataset directory.
    - port_num: Number of ports in the S-parameter files.
    - function_from_touchstone: An instance of FunctionFromTouchstone to parse the S-parameter files.
    - param_start_idx: The index to start reading geometric parameters from the filename.
    - force_regen: If True, regenerate the dataset even if the pickle file exists.
    - random_sample: If True, randomly sample 10 frequency points from the S-parameter data.

    Returns:
    - dataset: A list containing the S-parameter data.
    """
    # Check if the dataset path exists
    
    dataset = []
    caseid = 0
    for filename in tqdm(os.listdir(dataset_path), desc="Loading dataset", unit="file"):
        if filename.endswith(f".s{port_num}p"):
            file_path = os.path.join(dataset_path, filename) 
            
            # file name is {case_name_or_others}_{params}.snp
            # find 'HFSSDesign' and split by '_' (aborted)

            splits = filename.split('.')[0].split('_')
            splits = splits[param_start_idx:]  # remove the first part which is the case name

            # check if the pkl file exists
            pkl_file_path = os.path.join(dataset_path, f"case_{caseid}.pkl")

            if os.path.exists(pkl_file_path) and not force_regen:
                with open((pkl_file_path), 'rb') as f:
                    # reset the caseid to this one
                    data = pickle.load(f)    
                    data["caseid"] = caseid

            else:
                # generate a new dataset entry
                geo_param_vals = []
                while splits:
                    # assert float_p(splits[1]) >= 0, f"Invalid param: {splits[1]}"
                    geo_param_vals.append(float_p(splits[1])) # convert to float by replacing d to dot
                    splits = splits[2:]

                # load the touchstone file
                function_from_touchstone.parse(file_path)
                s_params = function_from_touchstone.evaluate_all()

                # generate sampled indices if random_sample is True
                if random_sample:
                    sampled_indices = np.random.choice(len(function_from_touchstone.freq), size=10, replace=False)
                else:
                    sampled_indices = None
                
                data = {
                    "caseid": caseid,
                    "geo_params": geo_param_vals,
                    "s_params": s_params,
                    "sampled_indices": sampled_indices
                }

                with open(pkl_file_path, 'wb') as f:
                    pickle.dump(data, f)

            dataset.append(data) # append the data to the dataset
            caseid += 1

    return dataset


def test_load_dataset():
    """
    Test the load_dataset function.
    """
    dataset_path = "/public/home/zhanghf/projects/vector/data/spiral_inductor_s_parameters_export"
    port_num = 2
    function_from_touchstone = FunctionFromTouchstone(port_num=port_num)  # Initialize with dummy freq
    param_start_idx = 4  # Adjust this based on your filename structure
    force_regen = False
    random_sample = False

    dataset = load_dataset(dataset_path, port_num, function_from_touchstone, param_start_idx, force_regen, random_sample)
    
    for data in dataset:
        print(f"Case ID: {data['caseid']}, Geo Params: {data['geo_params']}, Sampled Indices: {data['sampled_indices']}")
    
    print(f"Total cases loaded: {len(dataset)}")

    dataset_path = "/public/home/zhanghf/projects/vector/data/bandpass_coupled_line_filter_export"
    param_start_idx = 9
    dataset = load_dataset(dataset_path, port_num, function_from_touchstone, param_start_idx, force_regen, random_sample)

    for data in dataset:
        print(f"Case ID: {data['caseid']}, Geo Params: {data['geo_params']}, Sampled Indices: {data['sampled_indices']}")
    
    print(f"Total cases loaded: {len(dataset)}")