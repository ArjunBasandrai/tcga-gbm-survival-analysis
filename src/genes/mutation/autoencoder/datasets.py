import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ClinicalDataset(Dataset):
    def __init__(self, df, split="train", val_split=0.2, random_state=int(os.environ['PROJECT_SEED'])):
        self.df = df.drop(['patient_id', 'overall_survival'], axis=1)
        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        total_samples = self.df.shape[0]
        split_idx = int(total_samples * (1 - val_split))
        
        if split == "train":
            self.df = self.df.iloc[:split_idx]
        elif split == "val":
            self.df = self.df.iloc[split_idx:]
        elif split == "all":
            pass
        else:
            raise ValueError("split must be either 'train' or 'val'")

        self.scaler = StandardScaler()
        self.df = pd.DataFrame(self.scaler.fit_transform(self.df))
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data_item = self.df.iloc[idx].astype(np.float32)
        clinical_data = torch.tensor(data_item.values, dtype=torch.float32)
        return clinical_data
        
class MutationDataset(Dataset):
    def __init__(self, df, split="train", val_split=0.2, random_state=int(os.environ['PROJECT_SEED'])):
        self.df = df.drop(['patient_id'], axis=1)
        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        total_samples = self.df.shape[0]
        split_idx = int(total_samples * (1 - val_split))
        
        if split == "train":
            self.df = self.df.iloc[:split_idx]
        elif split == "val":
            self.df = self.df.iloc[split_idx:]
        elif split == "all":
            pass
        else:
            raise ValueError("split must be either 'train' or 'val'")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data_item = self.df.iloc[idx].astype(np.float32)
        mutation_data = torch.tensor(data_item.values, dtype=torch.float32)
        return mutation_data

def get_dataloaders(clinical_df, mutation_df, batch_size):
    train_dataset_clinical = ClinicalDataset(clinical_df, split="train")
    val_dataset_clinical = ClinicalDataset(clinical_df, split="val")
    clinical_train_loader = DataLoader(train_dataset_clinical, batch_size=batch_size, shuffle=True)
    clinical_val_loader = DataLoader(val_dataset_clinical, batch_size=batch_size, shuffle=False)

    train_dataset_mutation = MutationDataset(mutation_df, split="train")
    val_dataset_mutation = MutationDataset(mutation_df, split="val")
    mutation_train_loader = DataLoader(train_dataset_mutation, batch_size=batch_size, shuffle=True)
    mutation_val_loader = DataLoader(val_dataset_mutation, batch_size=batch_size, shuffle=False)

    clinical_dataset_all = ClinicalDataset(clinical_df, split="all")
    mutation_dataset_all = MutationDataset(mutation_df, split="all")

    return clinical_train_loader, clinical_val_loader, mutation_train_loader, mutation_val_loader, clinical_dataset_all, mutation_dataset_all