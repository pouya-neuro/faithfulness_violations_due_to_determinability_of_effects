import networkx as nx
import random
import numpy as np
import pandas as pd
from itertools import combinations,product
from scipy.stats import chi2_contingency,dirichlet, binom, geom
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

class CausalModel:
    def __init__(self, dag, supports, conditional_distributions, filters):
        self.dag = dag
        self.supports = supports
        self.conditional_distributions = conditional_distributions
        self.filters = filters
        self.variables = {}
        for node in dag.nodes:
            self.variables[node] = None

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
            parents = list(self.dag.predecessors(node))
            if not parents:  # Root node
                sample[node] = np.random.choice(self.supports[node], p=self.conditional_distributions[node])
            else:
                parent_values = tuple(sample[parent] for parent in parents)
                filtered_values = tuple(self.filters[(parent, node)](parent_values[i]) for i, parent in enumerate(parents))
                cond_probs = self.conditional_distributions[node][filtered_values]
                sample[node] = np.random.choice(list(cond_probs.keys()), p=list(cond_probs.values()))
        return sample

    def get_filter_partition(self, node, child):
        """Determine the partition of node's support when filtered for the edge node → child."""
        partition = {}
        for value in self.supports[node]:
            filtered_value = self.filters[(node, child)](value)
            if filtered_value not in partition:
                partition[filtered_value] = []
            partition[filtered_value].append(value)
        return partition

    def is_within_partition(self, conditional_support, partition):
        """Check if the conditional support is entirely within a single subset of the partition."""
        for subset in partition.values():
            if set(conditional_support).issubset(subset):
                return True
        return False

    def check_faithfulness_violations(self):
        violations_count = 0
        for node in self.dag.nodes:
            parents = list(self.dag.predecessors(node))
            if not parents:
                continue  # Skip root nodes
            filtered_parent_combinations = product(*[set(map(self.filters[(parent, node)], self.supports[parent])) for parent in parents])
            # Loop over each child of node
            for child in self.dag.successors(node):
                filter_partition = self.get_filter_partition(node, child)
                # Check conditional distributions of node given its parents
                all_conditions_within_partition = True
                for parent_values in filtered_parent_combinations:
                    conditional_support = [
                        value for value, prob in self.conditional_distributions[node][parent_values].items() if prob > 0
                    ]
                    if not self.is_within_partition(conditional_support, filter_partition):
                        all_conditions_within_partition = False
                        break
                if all_conditions_within_partition:
                    violations_count += 1
        return violations_count

def create_random_filter_function(support, mean_simplicity):
    range_size = binom(len(support)-1, 1-mean_simplicity).rvs()+1
    filter_range = np.random.choice(support, size=range_size, replace=False)
    mapping = {val: np.random.choice(filter_range) for val in support}
    def filter_func(value):
        return mapping[value]
    return filter_func

def create_random_dag(num_nodes, edge_prob=0.3, seed=None):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))
    
    if seed is not None:
        np.random.seed(seed)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                dag.add_edge(i, j)
    return dag

def create_causal_bayesian_network(num_variables, mean_support, mean_simplicity, mean_weakness, ensure_faithfulness_violation=False, ensure_no_faithfulness_violation=False):
    if  mean_support < 3:
        raise ValueError(f"mean_support must be greater or equal to 3.")
    if mean_simplicity < 0 or mean_simplicity > 1:
        raise ValueError("mean_simplicity must be between 0 and 1.")
    if mean_weakness < 0 or mean_weakness > 1:
        raise ValueError("mean_weakness must be between 0 and 1.")
    if ensure_faithfulness_violation and ensure_no_faithfulness_violation:
        raise ValueError("ensure_faithfulness_violation and ensure_no_faithfulness_violation cannot both be set to True.")
    while True:
        dag = create_random_dag(num_variables)
        supports = {}
        conditional_distributions = {}
        filters = {}
        # Sample the supports for each variable
        for node in dag.nodes:
            support_size = geom.rvs(1 / (mean_support-1)) + 1
            supports[node] = list(range(1, support_size+1))
            if not list(dag.predecessors(node)):  # Root node
                conditional_distributions[node] = dirichlet([1] * support_size).rvs()[0]
            else:
                for parent in dag.predecessors(node):
                    filters[(parent, node)] = create_random_filter_function(supports[parent], mean_simplicity)
                conditional_distributions[node] = {}
                filtered_parent_combinations = product(*[set(map(filters[(parent, node)], supports[parent])) for parent in dag.predecessors(node)])
                for filtered_values in filtered_parent_combinations:
                    cond_support_size = binom(len(supports[node])-1, 1-mean_weakness).rvs()+1
                    cond_support = np.random.choice(supports[node], size=cond_support_size, replace=False)
                    cond_probs = dirichlet([1] * cond_support_size).rvs()[0]
                    conditional_distribution = {val: 0 for val in supports[node]}
                    for i, val in enumerate(cond_support):
                        conditional_distribution[val] = cond_probs[i]
                    conditional_distributions[node][filtered_values] = conditional_distribution
        causal_model = CausalModel(dag, supports, conditional_distributions, filters) 
        if ensure_faithfulness_violation:
            if causal_model.check_faithfulness_violations() > 0:
                return causal_model
        if ensure_no_faithfulness_violation:
            if causal_model.check_faithfulness_violations() == 0:
                return causal_model
        elif not ensure_faithfulness_violation and not ensure_no_faithfulness_violation:
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
    description = ("This PDF gives an overview of the performance of the PC algorithm on discrete models "
                   "with faithfulness violations vs. discrete models without faithfulness violations. "
                   "The following table summarizes the accuracy of PC skeleton search:\n\n")
    # Calculate average accuracy for models with and without faithfulness violations
    violation_accuracies = [result["accuracy"] for result in results if result["has_violation"]]
    no_violation_accuracies = [result["accuracy"] for result in results if not result["has_violation"]]
    avg_violation_accuracy = np.mean(violation_accuracies) if violation_accuracies else 0
    avg_no_violation_accuracy = np.mean(no_violation_accuracies) if no_violation_accuracies else 0
    summary_text = (f"Average accuracy for models with faithfulness violations: {avg_violation_accuracy:.3f}\n"
                    f"Average accuracy for models without faithfulness violations: {avg_no_violation_accuracy:.3f}\n\n")
    ax.text(0.5, 0.9, description, ha='center', va='top', fontsize=12, wrap=True)
    ax.text(0.5, 0.8, summary_text, ha='center', va='top', fontsize=12, wrap=True)
    # Prepare table data
    table_data = [["Model Size", "Faithfulness Violation", "Accuracy"]]
    for result in results:
        table_data.append([result["model_size"], result["has_violation"], f"{result['accuracy']:.3f}"])
    # Create table
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    pdf.savefig(fig)
    plt.close(fig)

def run_script():
    seed = 42
    mean_support = 4  # Fixed mean support for endogenous variables
    num_examples_per_size = 2
    min_graph_size = 3
    max_graph_size = 10
    simplicity = 0.5
    weakness = 0.5

    # Set the random seed for reproducibility
    set_random_seed(seed)

    # Create directory for visualisations if it doesn't exist
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'visualisations'))
    os.makedirs(output_dir, exist_ok=True)

     # Create a PDF file to save all figures
    pdf_path = os.path.join(output_dir, 'weak_noise_discrete_models_examples.pdf')
    all_visualisation_results = []

    with PdfPages(pdf_path) as pdf:
        total_iterations = len([True, False]) * len(range(min_graph_size, max_graph_size+1)) * num_examples_per_size 
        with tqdm(total=total_iterations, desc="Generating visualisations") as pbar:
            for has_violation in [True, False]:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                section_title = "Models with faithfulness violations" if has_violation else "Models without faithfulness violations"
                ax.text(0.5, 0.5, section_title, ha='center', va='center', fontsize=16, fontweight='bold')
                pdf.savefig(fig)
                plt.close(fig)
                for num_nodes in range(min_graph_size, max_graph_size + 1):
                    for example_idx in range(num_examples_per_size):
                        # Create the causal Bayesian network
                        if has_violation:
                            causal_model = create_causal_bayesian_network(num_nodes, mean_support, simplicity, weakness, ensure_faithfulness_violation=True)
                        else:
                            causal_model = create_causal_bayesian_network(num_nodes, mean_support, simplicity, weakness, ensure_no_faithfulness_violation=True)
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
                            "simplicity": simplicity,
                            "weakness": weakness,
                            "has_violation": has_violation,
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

                        pbar.update(1)

        # Generate summary page for performance results
        generate_summary_page(pdf, all_visualisation_results)

    print(f"All visualisations have been saved to {pdf_path}")

    # Additional section for varying simplicity and weakness
    analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'analyses'))
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_pdf_path = os.path.join(analysis_dir, 'weak_noise_discrete_models_analysis_1.pdf')

    all_analysis_results = []

    total_iterations = 9 * (max_graph_size - min_graph_size + 1) * 5 * 2
    with tqdm(total=total_iterations, desc="perforrming analysis") as pbar:
        for simplicity in [0.1, 0.5, 0.9]:
            for weakness in [0.1, 0.5, 0.9]:
                for num_nodes in range(min_graph_size, max_graph_size + 1):
                    for example_idx in range(5):
                        for has_violation in [True, False]:
                            # Create the causal Bayesian network
                            if has_violation:
                                causal_model = create_causal_bayesian_network(num_nodes, mean_support, simplicity, weakness, ensure_faithfulness_violation=True)
                            else:
                                causal_model = create_causal_bayesian_network(num_nodes, mean_support, simplicity, weakness, ensure_no_faithfulness_violation=True)
                            data = causal_model.generate_data(10000)

                            # Get the true skeleton
                            true_skeleton = nx.DiGraph(causal_model.dag)

                            # Apply PC algorithm
                            estimated_skeleton, sep_set = pc_algorithm(data)

                            # Compare the true and estimated skeletons
                            accuracy = compare_skeletons(estimated_skeleton, true_skeleton)
                            # Store the results
                            all_analysis_results.append({
                                "simplicity": simplicity,
                                "weakness": weakness,
                                "model_size": num_nodes,
                                "has_violation": has_violation,
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

            violation_performance = {size: [] for size in range(min_graph_size, max_graph_size + 1)}
            no_violation_performance = {size: [] for size in range(min_graph_size, max_graph_size + 1)}

            for result in all_analysis_results:
                if result["has_violation"]:
                    violation_performance[result["model_size"]].append(result[metric])
                else:
                    no_violation_performance[result["model_size"]].append(result[metric])

            violation_avg = {size: np.mean(violation_performance[size]) for size in violation_performance}
            no_violation_avg = {size: np.mean(no_violation_performance[size]) for size in no_violation_performance}

            sizes = list(violation_avg.keys())
            violation_values = list(violation_avg.values())
            no_violation_values = list(no_violation_avg.values())

            ax.bar(sizes, violation_values, width=0.4, label='Faithfulness Violation', align='center')
            ax.bar(sizes, no_violation_values, width=0.4, label='No Faithfulness Violation', align='edge')

            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Summary table for analysis results
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        summary_table = [["Model Size", "Faithfulness Violation", "No Faithfulness Violation"]]
        for size in sizes:
            summary_table.append([
                size, 
                f"{violation_avg[size]:.3f}", 
                f"{no_violation_avg[size]:.3f}"
            ])
        table = ax.table(cellText=summary_table, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        pdf.savefig(fig)
        plt.close(fig)

        avg_violation_accuracy = np.mean(list(violation_avg.values()))
        avg_no_violation_accuracy = np.mean(list(no_violation_avg.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, "This PDF compares the performance of the skeleton search step of the\n"
                          "PC algorithm on discrete models with at least one faithfulness violations vs.\n"
                          "discrete models without faithfulness violations. The bar chart shows the average\n"
                          "accuracy across different model sizes (number of endogenous variables), aggregated\n"
                          "over 90 runs for each model size. In particular, 45 runs are made for models with\n"
                          "faithfulness violations, and 45 runs are made for models that strictly\n"
                          "have no faithfulness violations.\n"
                          f"\nAverage accuracy across all discrete models with faithfulness violations: {avg_violation_accuracy:.3f}"
                          f"\nAverage accuracy across all discrete models without faithfulness violations: {avg_no_violation_accuracy:.3f}",
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Performance analysis has been saved to {analysis_pdf_path}")

if __name__ == '__main__':
    run_script()
