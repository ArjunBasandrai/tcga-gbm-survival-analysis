import time

import os
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

import pickle as pkl
from tqdm import tqdm

import gseapy as gp
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from ...dataset.datasets import load_df

def _time_function(func, **kwargs):
    start = time.time()
    result = func(**kwargs)
    end = time.time()
    print(f"Time taken: {(end - start)/60:.2f} minutes")
    return result

def _get_top_go_pathways(all_genes):
    enr = gp.enrichr(
        gene_list=list(all_genes), 
        gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
        organism='human', 
        outdir=None
    )

    pathway_results = enr.results

    top_pathways = pathway_results.sort_values(by='Adjusted P-value').head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_pathways['Term'], x=-np.log10(top_pathways['Adjusted P-value']), palette="magma")

    plt.xlabel('-log(Adjusted P-value)')
    plt.ylabel('Pathway Name')
    plt.title('Top 10 Enriched Pathways (High vs. Low)')

    save_path = "results/genes/expression/feature_selection/gsva/go_enrichment/pathways.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
def _preprocess_data(rna_df, lvh_go_path):
    lvh_go_df = pd.read_csv(lvh_go_path).rename(columns={"p.adjust": "p_adj", "geneID": "genes"})

    lvh_go_df['genes'] = lvh_go_df['genes'].apply(lambda x: x.split('/'))
    all_genes = set(g for sublist in lvh_go_df['genes'] for g in sublist)

    _get_top_go_pathways(all_genes)

    rna_df = load_df("data/RNAseq2.csv").T
    rna_df.columns = rna_df.iloc[0]
    rna_df = rna_df.iloc[1:]
    rna_df = rna_df.astype(float)

    kegg_gene_sets = gp.get_library(name="KEGG_2021_Human")
    reactome_gene_sets = gp.get_library(name="Reactome_2022")

    combined_gene_sets = {**kegg_gene_sets, **reactome_gene_sets}

    go_genes = []
    for genes in lvh_go_df.genes.to_numpy():
        go_genes.extend(genes)

    go_genes = np.array(list(set(go_genes)))

    return rna_df, combined_gene_sets, go_genes

def _gsva(rna_df, combined_gene_sets,save_path):
    gsva_results = _time_function(gp.gsva, data=rna_df, gene_sets=combined_gene_sets, method='gsva', threads=4)
    gsva_results = gsva_results.res2d.pivot(index='Term', columns='Name', values='ES')
    
    print(gsva_results.head())
    gsva_results.to_csv(save_path, index=True)

def _mann_whitney_u_test_filtering(clinical_df, save_path, combined_gene_sets):
    gsva_results = pd.read_csv(save_path, index_col=0)
    rna_df = load_df("data/RNAseq2.csv")
    merged_df = rna_df.merge(
        clinical_df[['patient_id', 'status', 'overall_survival']],
        on="patient_id",
        how="inner"
    )
    merged_df['status'] = merged_df['status'].astype(int)
    merged_df['overall_survival'] = merged_df['overall_survival'].astype(int)

    q25 = merged_df['overall_survival'].quantile(0.25)
    q50 = merged_df['overall_survival'].quantile(0.50)
    q75 = merged_df['overall_survival'].quantile(0.75)

    merged_df['cluster'] = 2
    merged_df.loc[merged_df['overall_survival'] <= q25, 'cluster'] = 0
    merged_df.loc[(merged_df['overall_survival'] > q25) & (merged_df['overall_survival'] <= q50), 'cluster'] = 1
    merged_df.loc[(merged_df['overall_survival'] > q50) & (merged_df['overall_survival'] <= q75), 'cluster'] = 2
    merged_df.loc[merged_df['overall_survival'] > q75, 'cluster'] = 3

    metadata_df = merged_df.copy()
    metadata_df.set_index(metadata_df.columns[0], inplace=True)

    gsva_patients = set(gsva_results.columns)
    metadata_patients = set(metadata_df.index.values)

    common_patients = list(gsva_patients.intersection(metadata_patients))

    gsva_results = gsva_results[common_patients]

    metadata_df = metadata_df.loc[gsva_results.columns]

    high_samples = metadata_df[metadata_df["cluster"] == 3].index
    low_samples = metadata_df[metadata_df["cluster"] == 0].index

    stat_results = []
    for pathway in tqdm(gsva_results.index):
        high_scores = gsva_results.loc[pathway, high_samples]
        low_scores = gsva_results.loc[pathway, low_samples]

        high_scores = pd.to_numeric(high_scores, errors='coerce')
        low_scores = pd.to_numeric(low_scores, errors='coerce')

        stat, pval = mannwhitneyu(high_scores, low_scores, alternative='two-sided')
        stat_results.append((pathway, pval))

    stat_df = pd.DataFrame(stat_results, columns=["Pathway", "P-value"])
    stat_df["Adjusted P-value"] = multipletests(stat_df["P-value"], method='fdr_bh')[1]
    significant_stat_df = stat_df[stat_df["Adjusted P-value"] < 0.05].sort_values(by="Adjusted P-value")
    significant_gsva = gsva_results.loc[significant_stat_df.Pathway]

    print(significant_gsva.head())

    high_survival_gsva = significant_gsva[high_samples]
    low_survival_gsva = significant_gsva[low_samples]
    survival_gsva = pd.DataFrame({
        "High Survival" : high_survival_gsva.mean(axis=1),
        "Low Survival" : low_survival_gsva.mean(axis=1)
    })

    save_path = "results/genes/expression/feature_selection/gsva/gsva_survival_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    survival_gsva.to_csv(save_path)

    plot_save_path = "results/genes/expression/feature_selection/gsva/gsva_survival_results.png"
    plt.figure(figsize=(150,15))
    plt.bar(survival_gsva.index, survival_gsva['High Survival'], label="H", color="teal")
    plt.bar(survival_gsva.index, survival_gsva['Low Survival'], label="L", color="salmon")
    plt.legend()
    plt.xticks(rotation=90)
    plt.title("GSVA Pathway Scores for High and Low Survival Patients")
    plt.savefig(plot_save_path)

    top_genes = list()
    for pathway in significant_gsva.index:
        top_genes.extend(list(set(combined_gene_sets[pathway]).intersection(set(rna_df.columns[1:]))))
    top_genes = np.array(list(set(top_genes)))
    return top_genes, significant_gsva

def _log_rank_test_filtering(merged_gsva, pathway_cols):
    top_logrank_results = []
    figures = []

    high_color = "teal"
    low_color = "darkorange"

    for pathway in tqdm(pathway_cols):
        selected = merged_gsva[[pathway, "overall_survival", "status"]].copy()

        threshold = selected[pathway].median()
        high_scores = selected[selected[pathway] > threshold]
        low_scores = selected[selected[pathway] <= threshold]

        p_value = logrank_test(
            high_scores["overall_survival"], 
            low_scores["overall_survival"], 
            event_observed_A=high_scores["status"],
            event_observed_B=low_scores["status"]
        ).p_value

        if p_value < 0.05:
            top_logrank_results.append([pathway, p_value])
            
            kmf_high = KaplanMeierFitter()
            kmf_low = KaplanMeierFitter()

            fig, ax = plt.subplots(figsize=(10, 6))

            kmf_high.fit(high_scores["overall_survival"], high_scores["status"], label=f"High {pathway} expression")
            kmf_high.plot(ci_show=False, color=high_color, ax=ax)

            kmf_low.fit(low_scores["overall_survival"], low_scores["status"], label=f"Low {pathway} expression")
            kmf_low.plot(ci_show=False, color=low_color, ax=ax)

            ax.set_title(f"Kaplan-Meier Survival Curve for {pathway}\nLog-rank p = {p_value}")
            ax.set_xlabel("Time (Months)")
            ax.set_ylabel("Survival Probability")
            ax.legend()
            ax.grid(True)
            
            figures.append(fig)
            
            plt.close(fig)

    logrank_df = pd.DataFrame(top_logrank_results, columns=["Pathway", "P-value"])
    logrank_df["Figure"] = figures
    logrank_df = logrank_df.sort_values(by="P-value")

    return logrank_df

def run_gsva(clinical_df, rna_df, lvh_go_path):
    save_path = "results/genes/expression/feature_selection/gsva/gsva_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Preprocessing data...")
    rna_df, combined_gene_sets, go_genes = _preprocess_data(rna_df, lvh_go_path)

    if not os.path.exists(save_path):
        print("Running GSVA...")
        _gsva(rna_df, combined_gene_sets, save_path)
    
    print("Running Mann-Whitney U-Test...")
    top_genes, significant_gsva = _mann_whitney_u_test_filtering(clinical_df, save_path, combined_gene_sets)

    go_common_genes = np.array(list(set(go_genes).intersection(set(top_genes))))
    print(go_common_genes)

    significant_gsva = significant_gsva.T.reset_index().rename(columns={"index":"patient_id"})
    print(significant_gsva.head())
    pathway_cols = significant_gsva.columns[1:]

    merged_gsva = significant_gsva.merge(
        clinical_df[['patient_id', 'status', 'overall_survival']],
        on="patient_id",
        how="inner"
    )
    merged_gsva[['status', 'overall_survival']] = merged_gsva[['status', 'overall_survival']].astype(int)
    merged_gsva = merged_gsva.set_index('patient_id')

    logrank_df = _log_rank_test_filtering(merged_gsva, pathway_cols)
    total_gene_list = []

    for pathway in tqdm(logrank_df['Pathway']):
        total_gene_list.extend(combined_gene_sets[pathway])

    total_gene_list = np.array(list(set(total_gene_list)))
    common_genes = np.array(list(set(go_common_genes).intersection(set(total_gene_list))))
    print(common_genes)

    save_path = "results/genes/expression/feature_selection/common_genes.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(common_genes, f)