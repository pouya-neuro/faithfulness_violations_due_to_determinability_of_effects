import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from discrete_noise_script_1 import create_structural_causal_model, set_random_seed
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

def run_script():

    # Create directory for results if it doesn't exist
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'analyses'))
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, 'discrete_noise_discrete_models_analysis_2.pdf')

    seed = 42
    mean_support_noise_range = range(3,8)
    mean_support_endogenous_range = range(3,8)
    num_endogenous_variables_range = range(2, 11,2)
    num_samples_per_combination = 1000

    set_random_seed(seed)
    results = []

    total_iterations = len(mean_support_noise_range) * len(mean_support_endogenous_range) * len(num_endogenous_variables_range)
    with tqdm(total=total_iterations, desc="performing second analysis") as pbar:
        for mean_support_noise in mean_support_noise_range:
            for mean_support_endogenous in mean_support_endogenous_range:
                for num_endogenous_variables in num_endogenous_variables_range:
                    num_deterministic = 0
                    for _ in range(num_samples_per_combination):
                        causal_model = create_structural_causal_model(num_endogenous_variables, mean_support_noise, mean_support_endogenous)
                        if causal_model.check_determinism():
                            num_deterministic += 1
                    percentage_deterministic = (num_deterministic / num_samples_per_combination) * 100
                    results.append({
                        "mean_support_noise": mean_support_noise,
                        "mean_support_endogenous": mean_support_endogenous,
                        "num_endogenous_variables": num_endogenous_variables,
                        "percentage_deterministic": percentage_deterministic})
                    pbar.update(1)

    # Save the results and generate the PDF report
    with PdfPages(pdf_path) as pdf:
        # Summary Page
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis('off')
        summary_text = (f"This report summarizes the analysis of the prevalence of deterministic nodes in\n"
                        f"discrete causal models with discrete noise. The analysis was performed by varying\n"
                        f"the parameters mean_support_noise, mean_support_endogenous, and the number of\n"
                        f"endogenous variables. For each combination of parameters, 1000 models were sampled\n"
                        f"and the percentage of models with at least one deterministic node was recorded.")
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Combined data: Plot percentage of deterministic models as a function of mean_support_noise
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages = [np.mean([
                result["percentage_deterministic"]
                for result in results
                if result["mean_support_noise"] == mean_support_noise
            ]) for mean_support_noise in mean_support_noise_range]
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.plot(mean_support_noise_range, percentages, label='Combined data')
        ax.set_title('Percentage of Deterministic Models vs Mean Support Noise')
        ax.set_xlabel('Mean Support Noise')
        ax.set_ylabel('Percentage of Deterministic Models')
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Combined data: Plot percentage of deterministic models as a function of mean_support_endogenous
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages = [np.mean([
                result["percentage_deterministic"]
                for result in results
                if result["mean_support_endogenous"] == mean_support_endogenous
            ]) for mean_support_endogenous in mean_support_endogenous_range]
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.plot(mean_support_endogenous_range, percentages, label='Combined data')
        ax.set_title('Percentage of Deterministic Models vs Mean Support Endogenous')
        ax.set_xlabel('Mean Support Endogenous')
        ax.set_ylabel('Percentage of Deterministic Models')
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)

        # Plot percentage of deterministic models as a function of number of endogenous variables
        fig, ax = plt.subplots(figsize=(10, 6))
        average_percentages = [np.mean([
                result["percentage_deterministic"]
                for result in results
                if result["num_endogenous_variables"] == num_endogenous_variables
            ]) for num_endogenous_variables in num_endogenous_variables_range]
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.plot(num_endogenous_variables_range, average_percentages, label='Combined data')
        ax.set_title('Percentage of Deterministic Models vs Number of Endogenous Variables')
        ax.set_xlabel('Number of Endogenous Variables')
        ax.set_ylabel('Percentage of Deterministic Models')
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)

        # 3D plot of the percentage of deterministic models
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = np.array([result["mean_support_noise"] for result in results])
        y = np.array([result["mean_support_endogenous"] for result in results])
        z = np.array([result["num_endogenous_variables"] for result in results])
        c = np.array([result["percentage_deterministic"] for result in results])
        img = ax.scatter(x, y, z, c=c, cmap=plt.viridis())
        ax.set_xlabel('Mean Support Noise')
        ax.set_ylabel('Mean Support Endogenous')
        ax.set_zlabel('Number of Endogenous Variables')
        colorbar = fig.colorbar(img, ax=ax, label='Percentage of Deterministic Models')
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_title('3D Plot of Percentage of Deterministic Models')
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Analysis report has been saved to {pdf_path}")

if __name__ == '__main__':
    run_script()