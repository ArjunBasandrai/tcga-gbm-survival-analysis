import os
import numpy as np
import matplotlib.pyplot as plt

import gseapy as gp
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.model_selection import KFold

from tqdm import tqdm

def get_enriched_pathways(mutation_df):
    gene_cols = mutation_df.columns[1:].to_list()
    df = mutation_df[gene_cols]
    cluster_groups = df.groupby("cluster").mean()

    significance_level = 0.05

    enriched_pathways = []
    for cluster, row in cluster_groups.iterrows():
        gene_list = row[row > 0].index.tolist()
        gsea_results = gp.enrichr(gene_list=gene_list, 
                                gene_sets='KEGG_2019_Human', 
                                organism='human', 
                                outdir=None)
        enriched_pathways.append(gsea_results.results.query(f"`Adjusted P-value` < {significance_level}"))
        print(f"Top enriched pathways for Cluster {cluster}:")
        print(gsea_results.results[["Term", "Adjusted P-value"]].head())

    return enriched_pathways

def find_top_pathways(pathways_df, clinical_df, plot_path="results/pathways/coxph.png", cox_results="results/pathways/top_pathways.csv"):
    pathways_df = pathways_df.merge(
                    clinical_df[['patient_id', 'status']],
                    on="patient_id",
                    how="inner"
                )
    pathways_df['status'] = pathways_df['status'].astype(int)
    
    data = pathways_df.drop(['patient_id', 'cGMP-PKG signaling pathway'], axis=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    c_indices = []
    
    for train_idx, test_idx in tqdm(kf.split(data), total=kf.get_n_splits()):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx].copy()
        
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(train_df, duration_col='overall_survival', event_col='status')
        
        test_df['predicted'] = cph.predict_partial_hazard(test_df)
        
        c_idx = concordance_index(test_df['overall_survival'], -test_df['predicted'], test_df['status'])
        c_indices.append(c_idx)

    print(f"Mean C-index: {np.mean(c_indices):.4f}")
    print(f"Standard Deviation of C-index: {np.std(c_indices):.4f}")

    cph = CoxPHFitter(penalizer=0.03)
    cph.fit(data, duration_col='overall_survival', event_col='status')

    plt.figure(figsize=(35, 18))
    cph.plot()

    cph.check_assumptions(data, p_value_threshold=0.05)

    os.makedirs(os.path.dirname(cox_results), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    plt.savefig(plot_path)
    plt.show()

    cph.summary.loc[cph.summary["coef"].abs().sort_values(ascending=False).index].query("p < 0.05")[["coef", "exp(coef)", "p"]].to_csv(cox_results)