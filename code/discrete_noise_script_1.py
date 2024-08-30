import networkx as nx
import random
import numpy as np
import pandas as pd
from itertools import combinations, product
from scipy.stats import chi2_contingency, dirichlet, geom
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

class CausalModel:
    def __init__(self, dag, functions, supports, noise_supports, noise_distributions):
        self.dag = dag
        self.functions = functions
        self.supports = supports
        self.noise_supports = noise_supports
        self.noise_distributions = noise_distributions
        self.variables = {}
        self.noise_terms = {}
        for node in dag.nodes:
            self.variables[node] = None
            self.noise_terms[node] = None
    
    def generate_data(self, num_samples):
        data = {var: [] for var in self.variables}
        
        for _ in range(num_samples):
            sample = self.generate_sample()
            for var, value in sample.items():
                data[var].append(value)
                
        return pd.DataFrame(data)
    
    def generate_sample(self):
        sample = {}
        
        for node in nx.topological_sort(self.dag):
            noise_term = self.generate_noise_term(node)
            self.noise_terms[node] = noise_term
            if self.dag.in_degree(node) == 0:
                sample[node] = noise_term  # Exogenous variables take the value of their noise term
            else:
                parent_values = tuple(sample[parent] for parent in self.dag.predecessors(node))
                sample[node] = self.generate_endogenous_value(node, parent_values, noise_term)
                
        return sample
    
    def generate_noise_term(self, node):
        return np.random.choice(self.noise_supports[node], p=self.noise_distributions[node])
    
    def generate_endogenous_value(self, node, parent_values, noise_term):
        func = self.functions[node]
        return func(parent_values, noise_term)
    
    def check_determinism(self):
        deterministic_variables = []
        
        for node in self.dag.nodes:
            if self.dag.in_degree(node) > 0:  # Check if the node is endogenous
                parent_nodes = list(self.dag.predecessors(node))
                parent_supports = [self.supports[parent] for parent in parent_nodes]
                noise_support = self.noise_supports[node]
                
                is_deterministic = True
                
                # Iterate over every possible combination of parent values
                for parent_combination in product(*parent_supports):
                    values = set()
                    
                    # Iterate over every possible noise value
                    for noise in noise_support:
                        value = self.generate_endogenous_value(node, parent_combination, noise)
                        values.add(value)
                    
                    if len(values) > 1:
                        is_deterministic = False
                        break
                
                if is_deterministic:
                    deterministic_variables.append(node)
        
        return deterministic_variables

def create_random_function_endogenous(parent_supports, noise_support, child_support):
    joint_support = list(product(*parent_supports, noise_support))
    mapping = {input_tuple: np.random.choice(child_support) for input_tuple in joint_support}

    def func(parents, noise):
        return mapping[tuple(parents) + (noise,)]
    
    return func

def create_random_dag(num_nodes, edge_prob=0.3, seed=None):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))
    
    if seed is not None:
        np.random.seed(seed)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                dag.add_edge(i, j)
    
    return nx.relabel_nodes(dag, lambda x: f'V{x}')

def create_structural_causal_model(num_variables, mean_support_noise, mean_support_endogenous, ensure_deterministic=False,ensure_non_deterministic=False):
    if mean_support_noise < 3 or mean_support_endogenous < 3:
        raise ValueError(f"mean_support_noise and mean_support_endogenous must be greater or equal to 3.")
    if ensure_deterministic and ensure_non_deterministic:
        raise ValueError(f"ensure_deterministic and ensure_non_deterministic cannot both be simultaneously set to true.")
    while True:
        dag = create_random_dag(num_variables)
        supports = {}
        noise_supports = {}
        noise_distributions = {}
        functions = {}
        # First, initialize the supports and noise supports for all nodes
        for node in dag.nodes:
            # Determine the size of the support for the noise term
            noise_support_size = geom.rvs(1 / (mean_support_noise-1)) + 1
            noise_support = list(range(1, noise_support_size+1))
            noise_supports[node] = noise_support
            # Sample the Dirichlet distribution for the noise term's PMF
            noise_distribution = dirichlet.rvs([1] * noise_support_size)[0]
            noise_distributions[node] = noise_distribution
            if dag.in_degree(node) == 0:
                # For exogenous variables, the support is the same as the noise support
                supports[node] = noise_support
            else:
                # Determine the size of the support for the endogenous variable
                support_size = geom.rvs(1 / (mean_support_endogenous-1)) + 1
                supports[node] = list(range(1, support_size+1))
        # Now, initialize the functions for endogenous variables
        for node in dag.nodes:
            if dag.in_degree(node) > 0:
                parent_supports = [supports[parent] for parent in dag.predecessors(node)]
                functions[node] = create_random_function_endogenous(parent_supports, noise_supports[node], supports[node])
        causal_model = CausalModel(dag, functions, supports, noise_supports, noise_distributions)
        if ensure_deterministic:
            if causal_model.check_determinism():
                return causal_model
        if ensure_non_deterministic:
            if not causal_model.check_determinism():
                return causal_model
        elif (not ensure_deterministic) and (not ensure_non_deterministic):
            return causal_model

def pc_algorithm(data, alpha=0.05):
    variables = data.columns
    skeleton = nx.complete_graph(variables)  # Start with a complete undirected graph
    sep_set = {var: {} for var in variables}
    l = -1
    while True:
        l += 1
        cont = False
        edges = list(skeleton.edges())  # Create a list of edges to avoid modifying the graph while iterating
        for (i, j) in edges:
            adj_i = set(skeleton.neighbors(i)) - {j}
            if len(adj_i) >= l:
                for z in combinations(adj_i, l):
                    if is_independent(data, i, j, list(z), alpha):
                        skeleton.remove_edge(i, j)
                        sep_set[i][j] = z
                        sep_set[j][i] = z
                        cont = True
                        break
        if not cont:
            break
    return skeleton, sep_set

def is_independent(data, x, y, z, alpha):
    if len(z) == 0:
        # If z is empty, use chi-squared test of independence
        contingency_table = pd.crosstab(data[x], data[y])
        chi2, p, _, _ = chi2_contingency(contingency_table, correction=False)
        return p > alpha
    else:
        unique_z_vals = data[z].drop_duplicates()
        num_comparisons = len(unique_z_vals)
        adjusted_alpha = alpha / num_comparisons  # Bonferroni correction
        for val in unique_z_vals.itertuples(index=False):
            subset = data
            for zi, z_val in zip(z, val):
                subset = subset[subset[zi] == z_val]
            if subset.empty:
                continue
            contingency_table = pd.crosstab(subset[x], subset[y])
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                continue  # Not enough data to perform chi-squared test
            chi2, p, _, _ = chi2_contingency(contingency_table, correction=False)
            if p <= adjusted_alpha:
                return False
        return True

def compare_skeletons(estimated_skeleton, true_skeleton):
    nodes = set(estimated_skeleton.nodes())
    estimated_edges = set(estimated_skeleton.edges())
    true_edges = set(true_skeleton.edges())
    tp = len(estimated_edges & true_edges)
    fp = len(estimated_edges - true_edges)
    fn = len(true_edges - estimated_edges)
    tn = len([(i, j) for i in nodes for j in nodes if i != j and (i, j) not in estimated_edges and (i, j) not in true_edges])
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    return accuracy

def plot_dag(dag, true_skeleton, ax, title):
    # Assign a topological order layer to each node based on the true skeleton
    for layer, nodes in enumerate(nx.topological_generations(true_skeleton)):
        for node in nodes:
            dag.nodes[node]['layer'] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(dag, subset_key="layer")
    nx.draw(dag, pos, with_labels=True, ax=ax, node_size=500, node_color='skyblue', font_size=10)
    ax.set_title(title)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def generate_summary_page(pdf, results):
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    description = ("This PDF gives an overview of the performance of the PC algorithm on models "
                   "with deterministic variables vs. models without deterministic variables. "
                   "The following table summarizes the accuracy of PC skeleton search:\n\n")
    deterministic_accuracies = [result["accuracy"] for result in results if result["deterministic"]]
    non_deterministic_accuracies = [result["accuracy"] for result in results if not result["deterministic"]]
    avg_deterministic_accuracy = np.mean(deterministic_accuracies) if deterministic_accuracies else 0
    avg_non_deterministic_accuracy = np.mean(non_deterministic_accuracies) if non_deterministic_accuracies else 0
    summary_text = (f"Average accuracy for deterministic models: {avg_deterministic_accuracy:.3f}\n"
                    f"Average accuracy for non-deterministic models: {avg_non_deterministic_accuracy:.3f}\n\n")
    ax.text(0.5, 0.9, description, ha='center', va='top', fontsize=12, wrap=True)
    ax.text(0.5, 0.8, summary_text, ha='center', va='top', fontsize=12, wrap=True)
    table_data = [["Model Size", "Deterministic", "Accuracy"]]
    for result in results:
        table_data.append([result["model_size"], result["deterministic"], f"{result['accuracy']:.3f}"])
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    pdf.savefig(fig)
    plt.close(fig)

# In this section of the code, we generate visualisations showing how deterministic dependencies 
# for endogenous variables can affect the performance of the PC algorithm. 
def run_script():
    seed = 42
    mean_support_noise = 3
    mean_support_endogenous = 3
    num_examples_per_size = 2
    min_graph_size = 2
    max_graph_size = 10

    # set the random seed for reproducibility
    set_random_seed(seed)

    # Create directory for visualisations if it doesn't exist
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'visualisations'))
    os.makedirs(output_dir, exist_ok=True)

    # Create a PDF file to save all figures
    pdf_path = os.path.join(output_dir, 'discrete_noise_discrete_models_examples.pdf')
    all_visualisation_results = []

    with PdfPages(pdf_path) as pdf:
        # Determine the total number of iterations for the progress bar
        total_iterations = 2 * len(range(min_graph_size, max_graph_size + 1)) * num_examples_per_size
        with tqdm(total=total_iterations, desc="Generating visualisations") as pbar:
            for deterministic in [True, False]:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                section_title = "Models with some deterministic variables" if deterministic else "Models with only non-deterministic variables"
                ax.text(0.5, 0.5, section_title, ha='center', va='center', fontsize=16, fontweight='bold')
                pdf.savefig(fig)
                plt.close(fig)
                for num_nodes in range(min_graph_size, max_graph_size + 1):
                    for example_idx in range(num_examples_per_size):
                        # Create the causal model
                        if deterministic:
                            causal_model = create_structural_causal_model(num_nodes, mean_support_noise, mean_support_endogenous, ensure_deterministic=True)
                        else:
                            causal_model = create_structural_causal_model(num_nodes, mean_support_noise, mean_support_endogenous, ensure_non_deterministic=True)
                        data = causal_model.generate_data(10000)
                        
                        # Get the true skeleton
                        true_skeleton = nx.DiGraph(causal_model.dag)
                        
                        # Apply PC algorithm
                        estimated_skeleton, sep_set = pc_algorithm(data)
                        
                        # Compare the true and estimated skeletons
                        accuracy = compare_skeletons(estimated_skeleton, true_skeleton)
                        
                        # Store the results
                        all_visualisation_results.append({
                            "model_size": num_nodes,
                            "deterministic": deterministic,
                            "accuracy": accuracy
                        })
                        
                        # Visualization
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        
                        # Draw true skeleton
                        ax = axes[0]
                        ax.set_title("True DAG")
                        plot_dag(true_skeleton, true_skeleton, ax, "True DAG")
                        
                        # Draw estimated skeleton
                        ax = axes[1]
                        ax.set_title("Estimated Skeleton")
                        plot_dag(estimated_skeleton, true_skeleton, ax, "Estimated Skeleton")
                        
                        plt.tight_layout()
                        pdf.savefig(fig)  # Save the current figure to the PDF
                        plt.close(fig)  # Close the figure to free memory
                        
                        # Update the progress bar
                        pbar.update(1)

        # Generate summary page for performance results
        generate_summary_page(pdf, all_visualisation_results)

    print(f"All visualisations have been saved to {pdf_path}")

    # Additional section for looping over mean_support_noise and mean_support_endogenous
    analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'analyses'))
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_pdf_path = os.path.join(analysis_dir, 'discrete_noise_discrete_models_analysis_1.pdf')

    all_analysis_results = []

    total_iterations = 3 * 3 * (max_graph_size - min_graph_size + 1) * 5 * 2
    with tqdm(total=total_iterations, desc="performing analysis") as pbar:
        for mean_support_noise in range(3, 6):
            for mean_support_endogenous in range(3, 6):
                for num_nodes in range(min_graph_size, max_graph_size + 1):
                    for example_idx in range(5):
                        for deterministic in [True, False]:
                            # Create the causal model
                            if deterministic:
                                causal_model = create_structural_causal_model(num_nodes, mean_support_noise, mean_support_endogenous, ensure_deterministic=True)
                            else:
                                causal_model = create_structural_causal_model(num_nodes, mean_support_noise, mean_support_endogenous, ensure_non_deterministic=True)
                            data = causal_model.generate_data(10000)
                            
                            # Get the true skeleton
                            true_skeleton = nx.DiGraph(causal_model.dag)
                            
                            # Apply PC algorithm
                            estimated_skeleton, sep_set = pc_algorithm(data)
                            
                            # Compare the true and estimated skeletons
                            accuracy = compare_skeletons(estimated_skeleton, true_skeleton)
                            # Store the results
                            all_analysis_results.append({
                                "mean_support_noise": mean_support_noise,
                                "mean_support_endogenous": mean_support_endogenous,
                                "model_size": num_nodes,
                                "deterministic": deterministic,
                                "accuracy": accuracy
                            })
                            pbar.update(1)

    # Generate comparative performance analysis
    with PdfPages(analysis_pdf_path) as pdf:
        for metric in ["accuracy"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"PC Algorithm Performance - {metric.capitalize()}")
            ax.set_xlabel("Model Size")
            ax.set_ylabel(metric.capitalize())

            deterministic_performance = {size: [] for size in range(min_graph_size, max_graph_size + 1)}
            non_deterministic_performance = {size: [] for size in range(min_graph_size, max_graph_size + 1)}

            for result in all_analysis_results:
                if result["deterministic"]:
                    deterministic_performance[result["model_size"]].append(result[metric])
                else:
                    non_deterministic_performance[result["model_size"]].append(result[metric])

            deterministic_avg = {size: np.mean(deterministic_performance[size]) for size in deterministic_performance}
            non_deterministic_avg = {size: np.mean(non_deterministic_performance[size]) for size in non_deterministic_performance}

            sizes = list(deterministic_avg.keys())
            deterministic_values = list(deterministic_avg.values())
            non_deterministic_values = list(non_deterministic_avg.values())

            ax.bar(sizes, deterministic_values, width=0.4, label='Deterministic', align='center')
            ax.bar(sizes, non_deterministic_values, width=0.4, label='Non-Deterministic', align='edge')

            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Summary table for analysis results
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        summary_table = [["Model Size", "Deterministic", "Non-Deterministic"]]
        for size in sizes:
            summary_table.append([
                size, 
                f"{deterministic_avg[size]:.3f}", 
                f"{non_deterministic_avg[size]:.3f}"
            ])
        table = ax.table(cellText=summary_table, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        pdf.savefig(fig)
        plt.close(fig)

        avg_deterministic_accuracy = np.mean(list(deterministic_avg.values()))
        avg_non_deterministic_accuracy = np.mean(list(non_deterministic_avg.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, "This PDF compares the performance of the skeleton search step of the PC\n"
                          "algorithm on discrete models with at least one deterministic variables vs.\n"
                          "discrete models without deterministic variables. The bar chart shows the\n"
                          "average accuracy across different model sizes (number of endogenous variables),\n"
                          "aggregated over 90 runs for each model size. In particular, 45 runs are made\n"
                          "for models with deterministic endogenous variables, and 45 runs are made\n"
                          "for models that strictly have no deterministic endogenous variables.\n"
                          f"\nAverage accuracy acrosss all 405 deterministic models: {avg_deterministic_accuracy:.3f}"
                          f"\nAverage accuracy across all 405 non-deterministic models: {avg_non_deterministic_accuracy:.3f}",
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Performance analysis has been saved to {analysis_pdf_path}")

if __name__ == '__main__':
    run_script()