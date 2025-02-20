import os
import argparse
import pickle as pkl

os.environ['PROJECT_SEED'] = str(1834579290)

import pandas as pd

from src.dataset.clinical import load_clinical_data
from src.dataset.mutation import load_mutation_data

from src.autoencoder.datasets import get_dataloaders
from src.autoencoder.training.training import setup, train_autoencoder, cv_clinical_autoencoder, cv_mutation_autoencoder
from src.autoencoder.encoding import encode_clinical_data, encode_mutations_data
from src.autoencoder.chi2 import chi2_test

from src.pathways.pathways import get_enriched_pathways, find_top_pathways
from src.pathways.analyze_pathways import compare_pathways, analyze_pathway_a_b
from src.pathways.analyze_B import check_pten, find_top_genes

from src.config import Conf

def train_clinical_autoencoder(device, model, optimizer, loss_fn, scheduler, early_stopping, 
                               clinical_train_loader, clinical_val_loader, clinical_dataset_all, input_size):
    train_autoencoder(device, model, optimizer, loss_fn, scheduler, early_stopping, 
                      clinical_train_loader, clinical_val_loader, plot_path="results/clinical/loss.png")

    cv_clinical_autoencoder(device, loss_fn, early_stopping, clinical_dataset_all, input_size)

    encoded_clinical_df = encode_clinical_data(model, device, clinical_df)
    encoded_clinical_df.to_csv("processed/EncodedClinical.csv")


def train_mutation_autoencoder(device, model, optimizer, loss_fn, scheduler, early_stopping, 
                               mutation_train_loader, mutation_val_loader, mutation_dataset_all, input_size):
    train_autoencoder(device, model, optimizer, loss_fn, scheduler, early_stopping, 
                      mutation_train_loader, mutation_val_loader, plot_path="results/mutation/loss.png")

    cv_mutation_autoencoder(device, loss_fn, early_stopping, mutation_dataset_all, input_size)

    encoded_mutations_data = encode_mutations_data(model, device, mutation_df)

    encoded_mutations_data.to_csv("processed/EncodedMutation.csv")

def run_chi2_test(clinical_df, mutation_df):
    if os.path.exists("processed/EncodedMutation.csv") is False:
        raise ValueError("Please run mutation autoencoder first")

    chi2_test(clinical_df, mutation_df, "results/chi2/cluster_vs_survival.png")

def enrich_pathways(clinical_df, mutation_df):
    cox_results = "results/pathways/top_pathways.csv"

    if os.path.exists("processed/MutationClustered.csv") is False:
        raise ValueError("Please run the chi2 test first")
    
    if os.path.exists("processed/Pathways.csv") is False:
        print("Getting enriched pathways...")
        clustered_mutation_df = pd.read_csv("processed/MutationClustered.csv")
        enriched_pathways = get_enriched_pathways(clustered_mutation_df)
        pathways_df = pd.DataFrame(mutation_df['patient_id']).copy()
        for enriched_pathway in enriched_pathways:
            terms = enriched_pathway['Term']
            genes = enriched_pathway['Genes']
            for term, gene in zip(terms, genes):
                if term not in pathways_df.columns:
                    pathway_mutations = mutation_df[gene.split(";")].sum(axis=1)
                    pathways_df[term] = pathway_mutations
        pathways_df = pathways_df.merge(
            clinical_df[['patient_id', 'overall_survival']],
            on="patient_id",
            how="inner"
        )
        pathways_df.to_csv("processed/Pathways.csv")
    elif os.path.exists("results/pathways/top_pathways.csv") is False:
        print("Finding top pathways...")
        plot_path = "results/pathways/coxph.png"
        pathways_df = pd.read_csv("processed/Pathways.csv")
        find_top_pathways(pathways_df, clinical_df, plot_path=plot_path, cox_results=cox_results)
    else:
        print("Comparing pathways...")
        clustered_mutation_df = pd.read_csv("processed/MutationClustered.csv")
        important_pathways, enriched_pathways = compare_pathways(clustered_mutation_df, pd.read_csv(cox_results)['covariate'].to_list())

        pathway_B, pathway_A = important_pathways

        pathway_A_genes = enriched_pathways[0].query(f"Term == '{pathway_A}'")['Genes'].item().split(";")
        pathway_B_genes_0 = enriched_pathways[0].query(f"Term == '{pathway_B}'")['Genes'].item().split(";")
        pathway_B_genes_2 = enriched_pathways[2].query(f"Term == '{pathway_B}'")['Genes'].item().split(";")
        pathway_B_genes = list(set(pathway_B_genes_0 + pathway_B_genes_2))

        pkl.dump(pathway_B_genes, open("results/pathways/pathway_B_genes.pkl", "wb"))

        analyze_pathway_a_b(clinical_df, pathway_A, pathway_B)

def get_important_genes_from_pathway_B(mutation_df, clinical_df):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    mutation_df = mutation_df.merge(
        clinical_df[['patient_id', 'status', 'overall_survival']],
        on="patient_id",
        how="inner"
    )
    mutation_df['overall_survival'] = mutation_df['overall_survival'].astype(int)
    mutation_df['status'] = mutation_df['status'].astype(int)
    mutation_df = mutation_df.drop(columns=['patient_id'])

    pathway_B_genes = pkl.load(open("results/pathways/pathway_B_genes.pkl", "rb"))

    no_pten_verification = check_pten(mutation_df, pathway_B_genes)
    if no_pten_verification:
        print("Genes in Pathway B excluding PTEN are associated with survival")
        print("Finding top genes in Pathway B...")
        top_genes = find_top_genes(mutation_df, pathway_B_genes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Autoencoder")
    parser.add_argument("--clinical", action="store_true", help="Run Clinical Autoencoder")
    parser.add_argument("--mutation", action="store_true", help="Run Mutation Autoencoder")
    parser.add_argument("--chi2", action="store_true", help="Run Chi2 Test")
    parser.add_argument("--pathways", action="store_true", help="Get enriched pathways")
    parser.add_argument("--pathways_genes", action="store_true", help="Get important genes from pathway B")
    args = parser.parse_args()

    clinical_df, encoder = load_clinical_data("data/Clinical.csv")
    mutation_df = load_mutation_data("data/Mutation.csv")

    patients_to_remove = ['TCGA.DU.6392', 'TCGA.HT.8564']
    clinical_df = clinical_df[~clinical_df.patient_id.isin(patients_to_remove)]
    mutation_df = mutation_df[~mutation_df.patient_id.isin(patients_to_remove)]

    clinical_train_loader, clinical_val_loader, mutation_train_loader, mutation_val_loader, \
    clinical_dataset_all, mutation_dataset_all = get_dataloaders(clinical_df, mutation_df, batch_size=Conf.batch_size)

    device, models, optimizers, schedulers, loss_fns, early_stoppings = setup(clinical_df, mutation_df)

    if args.clinical:
        train_clinical_autoencoder(device, models[0], optimizers[0], loss_fns[0], schedulers[0], early_stoppings[0],
                                   clinical_train_loader, clinical_val_loader, clinical_dataset_all, 
                                   clinical_df.shape[1] - 2)

    if args.mutation:
        train_mutation_autoencoder(device, models[1], optimizers[1], loss_fns[1], schedulers[1], early_stoppings[1],
                                   mutation_train_loader, mutation_val_loader, mutation_dataset_all, 
                                   mutation_df.shape[1] - 1)

    if args.chi2:
        run_chi2_test(clinical_df, mutation_df)

    if args.pathways:
        enrich_pathways(clinical_df, mutation_df)

    if args.pathways_genes:
        get_important_genes_from_pathway_B(mutation_df, clinical_df)