import random
import numpy as np
from discrete_noise_script_1 import create_structural_causal_model, set_random_seed
from tqdm import tqdm

num_variables = 8
mean_support_noise = 2
mean_support_endogenous = 2
num_samples = 10000

def calculate_deterministic_percentage(num_variables, mean_support_noise, mean_support_endogenous, num_samples, seed=42):
    # Set the random seed for reproducibility
    set_random_seed(seed)
    
    num_deterministic = 0
    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            causal_model = create_structural_causal_model(num_variables, mean_support_noise, mean_support_endogenous)
            if causal_model.check_determinism():
                num_deterministic += 1
            pbar.update(1)
    
    percentage_deterministic = (num_deterministic / num_samples) * 100
    return percentage_deterministic


percentage = calculate_deterministic_percentage(num_variables, mean_support_noise, mean_support_endogenous, num_samples)
print(f"Percentage of models with at least one deterministic node: {percentage:.2f}%")
