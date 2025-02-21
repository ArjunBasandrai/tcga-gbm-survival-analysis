import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def encode_clinical_data(model, device, clinical_df):
    latent_representations = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(clinical_df.shape[0])):
            data = clinical_df.drop(['patient_id', 'overall_survival'], axis=1).iloc[i].astype(np.float32)
            data = torch.tensor(data.values, dtype=torch.float32).to(device)
            data = data.unsqueeze(0)

            _, latent_representation = model(data)
            latent_representations.append(latent_representation.cpu().numpy())

    latent_representations = np.vstack(latent_representations)

    latent_df = pd.DataFrame(latent_representations, columns=[f"latent_clinical_{i + 1}" for i in range(latent_representations.shape[1])])
    clinical_df = clinical_df.reset_index(drop=True)
    clinical_df = pd.concat([clinical_df, latent_df], axis=1)

    return clinical_df

def encode_mutations_data(model, device, mutation_df):
    latent_representations = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(mutation_df.shape[0])):
            data = mutation_df.drop(['patient_id'], axis=1).iloc[i].astype(np.float32)
            data = torch.tensor(data.values, dtype=torch.float32).to(device)
            data = data.unsqueeze(0)

            _, latent_representation = model(data)
            latent_representations.append(latent_representation.cpu().numpy())

    latent_representations = np.vstack(latent_representations)
    
    latent_df = pd.DataFrame(latent_representations, columns=[f"latent_mutation_{i+1}" for i in range(latent_representations.shape[1])])
    mutation_df = mutation_df.reset_index(drop=True)
    mutation_df = pd.concat([mutation_df, latent_df], axis=1)

    return mutation_df