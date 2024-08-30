import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from weak_noise_discrete_script_1 import create_causal_bayesian_network, set_random_seed
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

def run_script():
    # Create directory for results if it doesn't exist
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'analyses'))
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, 'weak_noise_discrete_models_analysis_2.pdf')

    seed = 42
    mean_simplicity_range = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mean_weakness_range = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mean_support = 4  # Fixed mean support for endogenous variables
    num_endogenous_variables_range = range(2, 11,2)
    num_samples_per_combination = 1000

    set_random_seed(seed)
    results = []

    total_iterations = len(mean_simplicity_range) * len(mean_weakness_range) * len(num_endogenous_variables_range)
    with tqdm(total=total_iterations, desc="Performing faithfulness violation analysis") as pbar:
        for mean_simplicity in mean_simplicity_range:
            for mean_weakness in mean_weakness_range:
                for num_endogenous_variables in num_endogenous_variables_range:
                    num_violations = 0
                    for _ in range(num_samples_per_combination):
                        causal_model = create_causal_bayesian_network(num_endogenous_variables, mean_support, mean_simplicity, mean_weakness)
                        if causal_model.check_faithfulness_violations() > 0:
                            num_violations += 1
                    percentage_violations = (num_violations / num_samples_per_combination) * 100
                    results.append({
                        "mean_simplicity": mean_simplicity,
                        "mean_weakness": mean_weakness,
                        "num_endogenous_variables": num_endogenous_variables,
                        "percentage_violations": percentage_violations
                    })
                    pbar.update(1)

    # Save the results and generate the PDF report
    with PdfPages(pdf_path) as pdf:
        # Summary Page
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis('off')
        summary_text = (f"This report summarizes the analysis of the prevalence of faithfulness violations\n"
                        f"in causal Bayesian networks. The analysis was performed by varying the\n"
                        f"parameters mean_simplicity, mean_weakness, and the number of\n"
                        f"endogenous variables. For each combination of parameters, 1000 models were sampled\n"
                        f"and the percentage of models with at least one faithfulness violation was recorded.")
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Combined data: Plot percentage of models with faithfulness violations as a function of mean_simplicity
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages = [np.mean([
                result["percentage_violations"]
                for result in results
                if result["mean_simplicity"] == mean_simplicity
            ]) for mean_simplicity in mean_simplicity_range]
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.plot(mean_simplicity_range, percentages, label='Combined data')
        ax.set_title('Percentage of Models with Faithfulness Violations vs Mean Simplicity')
        ax.set_xlabel('Mean Simplicity')
        ax.set_ylabel('Percentage of Models with Faithfulness Violations')
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Combined data: Plot percentage of models with faithfulness violations as a function of mean_weakness
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages = [np.mean([
                result["percentage_violations"]
                for result in results
                if result["mean_weakness"] == mean_weakness
            ]) for mean_weakness in mean_weakness_range]
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.plot(mean_weakness_range, percentages, label='Combined data')
        ax.set_title('Percentage of Models with Faithfulness Violations vs Mean Weakness')
        ax.set_xlabel('Mean Weakness')
        ax.set_ylabel('Percentage of Models with Faithfulness Violations')
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)

        # Plot percentage of models with faithfulness violations as a function of number of endogenous variables
        fig, ax = plt.subplots(figsize=(10, 6))
        average_percentages = [np.mean([
                result["percentage_violations"]
                for result in results
                if result["num_endogenous_variables"] == num_endogenous_variables
            ]) for num_endogenous_variables in num_endogenous_variables_range]
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.plot(num_endogenous_variables_range, average_percentages, label='Combined data')
        ax.set_title('Percentage of Models with Faithfulness Violations vs Number of Endogenous Variables')
        ax.set_xlabel('Number of Endogenous Variables')
        ax.set_ylabel('Percentage of Models with Faithfulness Violations')
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)

        # 3D plot of the percentage of models with faithfulness violations
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = np.array([result["mean_simplicity"] for result in results])
        y = np.array([result["mean_weakness"] for result in results])
        z = np.array([result["num_endogenous_variables"] for result in results])
        c = np.array([result["percentage_violations"] for result in results])
        img = ax.scatter(x, y, z, c=c, cmap=plt.viridis())
        ax.set_xlabel('Mean Simplicity')
        ax.set_ylabel('Mean Weakness')
        ax.set_zlabel('Number of Endogenous Variables')
        colorbar = fig.colorbar(img, ax=ax, label='Percentage of Models with Faithfulness Violations')
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_title('3D Plot of Percentage of Models with Faithfulness Violations')
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Analysis report has been saved to {pdf_path}")

if __name__ == '__main__':
    run_script()