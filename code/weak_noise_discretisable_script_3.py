import random
import numpy as np
from weak_noise_discretisable_script_1 import create_causal_bayesian_network, set_random_seed
from tqdm import tqdm

num_variables = 8
mean_simplicity = 0.3
mean_weakness = 0.3
num_samples = 10000

def calculate_faithfulness_violations_percentage(num_variables, mean_simplicity, mean_weakness, num_samples, seed=42):
    # Set the random seed for reproducibility
    set_random_seed(seed)
    num_violations = 0
    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            causal_model = create_causal_bayesian_network(num_variables, mean_simplicity, mean_weakness)
            if causal_model.check_faithfulness_violations():
                num_violations += 1
            pbar.update(1)
    
    percentage_faithfulness_violations = (num_violations / num_samples) * 100
    return percentage_faithfulness_violations

percentage = calculate_faithfulness_violations_percentage(num_variables, mean_simplicity, mean_weakness,num_samples)
print(f"Percentage of models with at least one faithfulness violation: {percentage:.2f}%")


