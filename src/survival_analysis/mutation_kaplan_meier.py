import os
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def kaplan_meier_analysis(df: pd.DataFrame, output_dir: str, label_mappings: dict) -> None:
    significant_genes = []
    p_value_threshold = 0.1
    min_group_size = 80

    pbar = tqdm(df.columns[1:-3])
    for gene in pbar:
        pbar.set_description(f"Processing Gene {gene}")
        mutated = df[df[gene] == 1]
        non_mutated = df[df[gene] == 0]

        if len(mutated) < min_group_size or len(non_mutated) < min_group_size:
            continue

        results_logrank = logrank_test(
            mutated["overall_survival"], non_mutated["overall_survival"],
            event_observed_A=mutated["status"], event_observed_B=non_mutated["status"]
        )

        if results_logrank.p_value < p_value_threshold:
            significant_genes.append({
                "gene": gene,
                "mutated_count": len(mutated),
                "non_mutated_count": len(non_mutated),
                "mutated_mean_survival (days)": round(mutated['overall_survival'].mean(), 2),
                "non_mutated_mean_survival (days)": round(non_mutated['overall_survival'].mean(), 2),
                "p_value": results_logrank.p_value,
            })
        
            kmf = KaplanMeierFitter()
            
            plt.figure(figsize=(20, 10))
            kmf.fit(mutated["overall_survival"], event_observed=mutated["status"], label="Mutated")
            kmf.plot_survival_function(color="teal")
            
            kmf.fit(non_mutated["overall_survival"], event_observed=non_mutated["status"], label="Non-Mutated")
            kmf.plot_survival_function(color="maroon")

            _output_dir = os.path.join(output_dir, gene)
            os.makedirs(_output_dir, exist_ok=True)

            plt.title(f"Kaplan-Meier Survival Curve for {gene} Mutation")
            plt.xlabel("Time (days)")
            plt.ylabel("Survival Probability")
            plt.legend()
            plt.grid(True)
                            
            plot_filename = os.path.join(_output_dir, f"{gene}_curve.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close() 

            _output_dir = os.path.join(_output_dir, "types")

            type_count = {}

            for unique in df['histological_type'].unique():
                label = list(label_mappings['histological_type'].items())[unique][0]

                type_count[label] = len(df[df['histological_type'] == unique])
                
                mutated_hist = mutated[mutated['histological_type'] == unique]
                non_mutated_hist = non_mutated[non_mutated['histological_type'] == unique]

                if len(mutated_hist) > 10 and len(non_mutated_hist) > 10:
                    os.makedirs(_output_dir, exist_ok=True)

                    plt.figure(figsize=(20, 10))

                    kmf.fit(mutated_hist["overall_survival"], event_observed=mutated_hist["status"], label=f"Mutated {label}")
                    kmf.plot_survival_function(color="teal")
                
                    kmf.fit(non_mutated_hist["overall_survival"], event_observed=non_mutated_hist["status"], label=f"Non-Mutated {label}")
                    kmf.plot_survival_function(color="maroon")
                
                    plt.title(f"Kaplan-Meier Survival Curve for {gene} Mutation for \"{label}\" type")
                    plt.xlabel("Time (days)")
                    plt.ylabel("Survival Probability")
                    plt.legend()
                    plt.grid(True)
                                                
                    plot_filename = os.path.join(_output_dir, f"{gene}_{label}_curve.png")
                    plt.savefig(plot_filename, dpi=300)
                    plt.close() 

            pd.DataFrame([type_count]).to_csv(os.path.join(output_dir, gene, f"{gene}_type_counts.csv"))

    significant_genes_df = pd.DataFrame(significant_genes)
    significant_genes_df.to_csv(os.path.join(output_dir, "significant_genes.csv"))