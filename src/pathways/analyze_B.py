import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

import pickle as pkl

def check_pten(mutation_df, pathway_B_genes):
    pathway_B_noPTEN = list(set(pathway_B_genes) - set(['PTEN']))
    PTEN = ['PTEN']

    no_pten_mutated = mutation_df[mutation_df[pathway_B_noPTEN].sum(axis=1) > 0]
    no_pten_non_mutated = mutation_df[mutation_df[pathway_B_noPTEN].sum(axis=1) == 0]
    pten_mutated = mutation_df[mutation_df[PTEN].sum(axis=1) > 0]
    pten_non_mutated = mutation_df[mutation_df[PTEN].sum(axis=1) == 0]

    kmf_no_pten_mutated = KaplanMeierFitter()
    kmf_no_pten_non_mutated = KaplanMeierFitter()
    kmf_pten_mutated = KaplanMeierFitter()
    kmf_pten_non_mutated = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))

    kmf_no_pten_mutated.fit(no_pten_mutated["overall_survival"], no_pten_mutated["status"], label="Pathway B (No PTEN) Mutated")
    kmf_no_pten_mutated.plot_survival_function()

    kmf_no_pten_non_mutated.fit(no_pten_non_mutated["overall_survival"], no_pten_non_mutated["status"], label="Pathway B (No PTEN) Non-Mutated")
    kmf_no_pten_non_mutated.plot_survival_function()

    kmf_pten_mutated.fit(pten_mutated["overall_survival"], pten_mutated["status"], label="PTEN Mutated")
    kmf_pten_mutated.plot_survival_function()

    kmf_pten_non_mutated.fit(pten_non_mutated["overall_survival"], pten_non_mutated["status"], label="PTEN Non-Mutated")
    kmf_pten_non_mutated.plot_survival_function()
    
    save_path = "results/pathways/pten/pten_analysis.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.title("Kaplan-Meier Survival Curves")
    plt.xlabel("Time (Days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

    logrank_no_pten = logrank_test(
        no_pten_mutated["overall_survival"], 
        no_pten_non_mutated["overall_survival"], 
        event_observed_A=no_pten_mutated["status"], 
        event_observed_B=no_pten_non_mutated["status"]
    )

    logrank_pten = logrank_test(
        pten_mutated["overall_survival"], 
        pten_non_mutated["overall_survival"], 
        event_observed_A=pten_mutated["status"], 
        event_observed_B=pten_non_mutated["status"]
    )

    print(f"Log-Rank p-value (Pathway B No PTEN Mutation vs No Mutation): {logrank_no_pten.p_value}")
    print(f"Log-Rank p-value (PTEN Mutation vs No Mutation): {logrank_pten.p_value}\n")

    return True if logrank_no_pten.p_value < 0.05 else False


def find_top_genes(mutation_df, pathway_B_genes):
    cox_data = mutation_df.copy()
    cox_data = cox_data[pathway_B_genes + ["overall_survival", "status"]]

    cox_data['MLXIPL'] = pd.cut(cox_data['MLXIPL'], bins=3, labels=[0,1,2]).astype("category")

    print("Training Cox Proportional Hazards model on all pathway B genes...")
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data, duration_col="overall_survival", event_col="status", strata=['PIK3CA', 'MLXIPL', 'IRS1'])
    print("Finished training")

    cph.check_assumptions(cox_data, p_value_threshold=0.05)

    top_pathway_genes = cph.summary.query("p < 0.05").index.to_list()

    print("Keeping only unique rows")
    cox_data = cox_data.drop_duplicates(subset=top_pathway_genes).reset_index(drop=True)

    print("Calculating C-index...")
    c_index = concordance_index(cox_data['overall_survival'], 
                            -cph.predict_partial_hazard(cox_data), 
                            cox_data['status'])
    print(f"C-index: {c_index:.4f}")


    print("\nPercentage of patients with mutation in top genes:")
    for gene in top_pathway_genes:
        print(f"{gene} : {mutation_df[gene].sum() / len(mutation_df) * 100:.2f}%")
    print()

    cox_data = mutation_df.copy()
    cox_data = cox_data[top_pathway_genes + ["overall_survival", "status"]]

    print("Training Cox Proportional Hazards model on top pathway B genes...")
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data, duration_col="overall_survival", event_col="status")
    print("Finished training")

    cph.check_assumptions(cox_data, p_value_threshold=0.05)

    top_genes = cph.summary.query("p < 0.05").index.to_list()

    print("Keeping only unique rows")
    cox_data = cox_data.drop_duplicates(subset=top_genes).reset_index(drop=True)

    print("Calculating C-index...")
    c_index = concordance_index(cox_data['overall_survival'], 
                            -cph.predict_partial_hazard(cox_data), 
                            cox_data['status'])
    print(f"C-index: {c_index:.4f}")

    with open('models/cox_model.pkl', 'wb') as f:
        pkl.dump(cph, f)

    print("\nTop pathway B genes affecting overall survival:")
    for gene in top_genes:
        print(f"{gene}", end=" ")
    return top_genes