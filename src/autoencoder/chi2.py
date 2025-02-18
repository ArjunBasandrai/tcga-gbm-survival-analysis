import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
    
from scipy.stats import chi2_contingency

import os

import seaborn as sns
import matplotlib.pyplot as plt

from ..config import Conf

def _get_latent_mutation_df(clinical_df):
    latent_mutation_df = pd.read_csv("processed/EncodedMutation.csv")
    latent_columns = latent_mutation_df.columns[-Conf.latent_dim:].to_list()
    latent_mutation_df = latent_mutation_df[latent_columns +  ['patient_id']]

    latent_mutation_df = latent_mutation_df.merge(clinical_df[['patient_id', 'overall_survival']], on='patient_id', how='inner')

    return latent_mutation_df, latent_columns

def _scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def _kmeans_cluster_latent(clinical_df):
    latent_mutation_df, latent_cols = _get_latent_mutation_df(clinical_df)
    X = latent_mutation_df[latent_cols].values

    X_scaled = _scale_data(X)

    kmeans = KMeans(n_clusters=3, random_state=999, n_init=10)
    latent_mutation_df['cluster'] = kmeans.fit_predict(X_scaled)

    latent_mutation_df['survival_bin'], bin_edges = pd.qcut(
        latent_mutation_df['overall_survival'], q=3, labels=["Low", "Medium", "High"], retbins=True
    )

    print("Bin Boundaries:", bin_edges)

    return latent_mutation_df

def chi2_test(clinical_df, mutation_df, plot_path):
    latent_mutation_df = _kmeans_cluster_latent(clinical_df)

    contingency_table = pd.crosstab(latent_mutation_df['cluster'], latent_mutation_df['survival_bin'])

    sns.heatmap(contingency_table, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Survival Bin")
    plt.ylabel("Cluster")
    plt.title("Cluster vs Binned Survival")

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)

    plt.show()

    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Statistic: {chi2:.2f}, p-value: {p}")

    if p < 0.05:
        mutation_df = mutation_df.merge(latent_mutation_df[['patient_id', 'cluster']], on='patient_id', how='inner')
        mutation_df.to_csv("processed/MutationClustered.csv", index=False)
    else:
        print("No significant difference between clusters and survival")