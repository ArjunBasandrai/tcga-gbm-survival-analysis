import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import gseapy as gp
from collections import defaultdict

def compare_pathways(mutation_df, top_pathways):
    cluster_groups = mutation_df.drop(columns=['patient_id']).groupby("cluster").mean()
    significance_level = 0.05

    enriched_pathways = []
    for _, row in cluster_groups.iterrows():
        gene_list = row[row > 0].index.tolist()
        gsea_results = gp.enrichr(gene_list=gene_list, 
                                gene_sets='KEGG_2019_Human', 
                                organism='human', 
                                outdir=None)
        enriched_pathways.append(gsea_results.results.query(f"`Adjusted P-value` < {significance_level}"))

    filtered_pathways = []
    for enriched_pathway in enriched_pathways:
        filtered_pathways.append(enriched_pathway[enriched_pathway["Term"].isin(top_pathways)]['Term'].to_list())

    membership = defaultdict(list)

    for i, sub_list in enumerate(filtered_pathways, start=1):
        for element in sub_list:
            membership[element].append(i - 1)

    important_elements = []
    for element, which_lists in membership.items():
        if len(which_lists) != len(filtered_pathways):
            print(f"Pathway {element} is in cluster(s): {which_lists}")  
            important_elements.append(element)

    print(f"Important elements: {important_elements}")
    return important_elements, enriched_pathways

def km_analysis(pathways_df, pathway_A, pathway_B):
    df = pathways_df.copy().drop(['patient_id'], axis=1)
    df['A_mut'] = df[pathway_A] > 0
    df['B_mut'] = df[pathway_B] > 0
    df['group'] = np.select(
        [
            (df['A_mut']) & (~df['B_mut']),
            (~df['A_mut']) & (df['B_mut']),
            (df['A_mut']) & (df['B_mut'])
        ],
        ['A_only', 'B_only', 'A_and_B'],
        default='Neither'
    )

    df_a_only = df[df['group'] == 'A_only']
    df_b_only = df[df['group'] == 'B_only']
    df_a_b = df[df['group'] == 'A_and_B']
    df_neither = df[df['group'] == 'Neither']

    kmf_a_only = KaplanMeierFitter().fit(
        df_a_only['overall_survival'], df_a_only['status'], label='A_only'
    )
    kmf_b_only = KaplanMeierFitter().fit(
        df_b_only['overall_survival'], df_b_only['status'], label='B_only'
    )
    kmf_a_b = KaplanMeierFitter().fit(
        df_a_b['overall_survival'], df_a_b['status'], label='A_and_B'
    )
    kmf_neither = KaplanMeierFitter().fit(
        df_neither['overall_survival'], df_neither['status'], label='Neither'
    )

    plt.figure(figsize=(15,10))
    ax = kmf_a_only.plot()
    kmf_b_only.plot(ax=ax)
    kmf_a_b.plot(ax=ax)
    kmf_neither.plot(ax=ax)
    plt.title('Kaplan-Meier Curves for Pathway A, Pathway B')
    plt.savefig("results/genes/mutation/pathways/pathway_A_B_km_curves.png")
    plt.show()

    res_a_neither = logrank_test(
        df_a_only['overall_survival'], df_neither['overall_survival'],
        event_observed_A=df_a_only['status'],
        event_observed_B=df_neither['status']
    )
    res_b_neither = logrank_test(
        df_b_only['overall_survival'], df_neither['overall_survival'],
        event_observed_A=df_b_only['status'],
        event_observed_B=df_neither['status']
    )
    res_a_b = logrank_test(
        df_a_only['overall_survival'], df_b_only['overall_survival'],
        event_observed_A=df_a_only['status'],
        event_observed_B=df_b_only['status']
    )
    res_a_ab = logrank_test(
        df_a_only['overall_survival'], df_a_b['overall_survival'],
        event_observed_A=df_a_only['status'],
        event_observed_B=df_a_b['status']
    )
    res_b_ab = logrank_test(
        df_b_only['overall_survival'], df_a_b['overall_survival'],
        event_observed_A=df_b_only['status'],
        event_observed_B=df_a_b['status']
    )
    res_ab_neither = logrank_test(
        df_a_b['overall_survival'], df_neither['overall_survival'],
        event_observed_A=df_a_b['status'],
        event_observed_B=df_neither['status']
    )

    print('A vs Neither p-value:', res_a_neither.p_value)
    print('B vs Neither p-value:', res_b_neither.p_value)
    print('A vs B p-value:', res_a_b.p_value)
    print('A vs A_and_B p-value:', res_a_ab.p_value)
    print('B vs A_and_B p-value:', res_b_ab.p_value)
    print('A_and_B vs Neither p-value:', res_ab_neither.p_value)

    log_rank_results = pd.DataFrame({
        'Comparison': ['A vs Neither', 'B vs Neither', 'A vs B', 'A vs A_and_B', 'B vs A_and_B', 'A_and_B vs Neither'],
        'p_value': [res_a_neither.p_value, res_b_neither.p_value, res_a_b.p_value, res_a_ab.p_value, res_b_ab.p_value, res_ab_neither.p_value]
    })
    
    return log_rank_results

def analyze_pathway_a_b(clinical_df, pathway_A, pathway_B):
    kmf_A = KaplanMeierFitter()
    kmf_B = KaplanMeierFitter()
    kmf_A_B = KaplanMeierFitter()

    pathways_df = pd.read_csv("processed/Pathways.csv").drop(
                        ['Unnamed: 0'], axis=1
                    ).merge(
                        clinical_df[['patient_id', 'status']],
                        on="patient_id",
                        how="inner",
                    )
    pathways_df['status'] = pathways_df['status'].astype(int)
    log_rank_results = km_analysis(pathways_df, pathway_A, pathway_B)
    print(log_rank_results.query("p_value < 0.05"))