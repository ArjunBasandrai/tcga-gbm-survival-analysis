import os

os.environ['PROJECT_SEED'] = str(1834579290)

from src.dataset.clinical import load_clinical_data
from src.dataset.mutation import load_mutation_data

from src.autoencoder.datasets import get_dataloaders
from src.autoencoder.training.training import setup, train_autoencoder, cv_clinical_autoencoder, cv_mutation_autoencoder
from src.autoencoder.encoding import encode_clinical_data, encode_mutations_data
from src.autoencoder.chi2 import chi2_test

from src.config import Conf

clinical_df, encoder = load_clinical_data("data/Clinical.csv")
mutation_df = load_mutation_data("data/Mutation.csv")

clinical_df = clinical_df[~clinical_df.patient_id.isin(['TCGA.DU.6392', 'TCGA.HT.8564'])]
mutation_df = mutation_df[~mutation_df.patient_id.isin(['TCGA.DU.6392', 'TCGA.HT.8564'])]

clinical_train_loader, clinical_val_loader, mutation_train_loader, mutation_val_loader, clinical_dataset_all, mutation_dataset_all = get_dataloaders(clinical_df, mutation_df, batch_size=Conf.batch_size)

device, models, optimizers, schedulers, loss_fns, early_stoppings = setup(clinical_df, mutation_df)

# train_autoencoder(device, models[0], optimizers[0], loss_fns[0], schedulers[0], early_stoppings[0], clinical_train_loader, clinical_val_loader, plot_path="results/clinical/loss.png")
# # cv_clinical_autoencoder(device, loss_fns[0], early_stoppings[0], clinical_dataset_all, clinical_df.shape[1] - 2)

# encoded_clinical_df = encode_clinical_data(models[0], device, clinical_df)
# print(encoded_clinical_df.head())

# encoded_clinical_df.to_csv("data/EncodedClinical.csv")

train_autoencoder(device, models[1], optimizers[1], loss_fns[1], schedulers[1], early_stoppings[1], mutation_train_loader, mutation_val_loader, plot_path="results/mutation/loss.png")
cv_mutation_autoencoder(device, loss_fns[1], early_stoppings[1], mutation_dataset_all, mutation_df.shape[1] - 1)

encoded_mutations_data = encode_mutations_data(models[1], device, mutation_df)
print(encoded_mutations_data.head())

encoded_mutations_data.to_csv("data/EncodedMutation.csv")

chi2_test(clinical_df, "results/chi2/cluster_vs_survival.png")