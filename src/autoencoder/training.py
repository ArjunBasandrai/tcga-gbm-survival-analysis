import torch
from torch import nn, optim

import os

from ..config import Conf
from .models import ClinicalAE, MutationAE
from .trainers import train_model, validate_model, plot_losses
from .cv import k_fold_autoencoder_training

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.path = path
        self.best_epoch = -1

    def reset(self):
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = -1
        
    def __call__(self, epoch, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(self.path, weights_only=True))
                return True
        return False
    
def setup(clinical_df, mutation_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clinical_ae = ClinicalAE(clinical_df.shape[1] - 2, latent_dim=Conf.latent_dim).to(device)
    mutation_ae = MutationAE(mutation_df.shape[1] - 1, latent_dim=Conf.latent_dim).to(device)

    optimizer_clinical = optim.Adam(clinical_ae.parameters(), lr=Conf.clinical_lr, weight_decay=Conf.clinical_weight_decay)
    optimizer_mutation = optim.Adam(mutation_ae.parameters(), lr=Conf.mutation_lr, weight_decay=Conf.mutation_weight_decay)

    scheduler_clinical = optim.lr_scheduler.CosineAnnealingLR(optimizer_clinical, T_max=50, eta_min=1e-6)
    scheduler_mutation = optim.lr_scheduler.CosineAnnealingLR(optimizer_mutation, T_max=50, eta_min=1e-6)

    loss_fn_clinical = nn.MSELoss()
    loss_fn_mutation = nn.BCELoss()

    early_stopping_clinical = EarlyStopping(patience=10, min_delta=0.0001, path="checkpoints/best_clinical_ae.pth")
    early_stopping_mutation = EarlyStopping(patience=10, min_delta=0.0001, path="checkpoints/best_mutation_ae.pth")

    models = [clinical_ae, mutation_ae]
    optimizers = [optimizer_clinical, optimizer_mutation]
    schedulers = [scheduler_clinical, scheduler_mutation]
    loss_fns = [loss_fn_clinical, loss_fn_mutation]
    early_stoppings = [early_stopping_clinical, early_stopping_mutation]

    return device, models, optimizers, schedulers, loss_fns, early_stoppings

def train_autoencoder(device, model, optimizer, loss_fn, scheduler, early_stopping, train_loader, val_loader, plot_path="results/loss.png"):
    train_loss_history = []
    val_loss_history = []

    for epoch in range(Conf.num_epochs):
        train_loss = train_model(model, optimizer, loss_fn, train_loader, device)
        val_loss = validate_model(model, loss_fn, val_loader, device)

        scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{Conf.num_epochs}")
        print(f"  Train -> Train Loss: {train_loss:.4f}")
        print(f"  Val   -> Val Loss: {val_loss:.4f}")

        if early_stopping(epoch, val_loss, model):
            print(f"\nEarly Stopping triggered with Best Epoch {early_stopping.best_epoch} "
                f"and Best Validation Loss: {early_stopping.best_loss:.4f}")
            break

    plot_losses(train_loss_history, val_loss_history, plot_path, clip=False)

def cv_clinical_autoencoder(device, loss_fn, early_stopping, dataset, input_size):
    fold_results = k_fold_autoencoder_training(
        model_ae=ClinicalAE,
        optimizer=None,
        scheduler=None,
        early_stopping=early_stopping,
        device=device,
        dataset=dataset,
        train_autoencoder_fn=train_model,
        validate_autoencoder_fn=validate_model,
        plot_losses_fn=plot_losses,
        loss_fn=loss_fn,
        lr=Conf.clinical_lr,
        weight_decay=Conf.clinical_weight_decay,
        training_graph_path="results/clinical/cv/clinical_autoencoder_cv",
        latent_dim=Conf.latent_dim,
        input_size=input_size,
        n_splits=5,
        num_epochs=Conf.num_epochs,
        batch_size=Conf.batch_size,
        shuffle_dataset=True,
        random_state=int(os.environ['PROJECT_SEED'])
    )

    for fold_info in fold_results:
        print(f"Fold {fold_info['fold']} results:")
        print(f"  Best Epoch: {fold_info['best_epoch']}")
        print(f"  Best Validation Loss: {fold_info['best_val_loss']:.4f}")
        print("-" * 40)

def cv_mutation_autoencoder(device, loss_fn, early_stopping, dataset, input_size):
    fold_results = k_fold_autoencoder_training(
        model_ae=MutationAE,
        optimizer=None,
        scheduler=None,
        early_stopping=early_stopping,
        device=device,
        dataset=dataset,
        train_autoencoder_fn=train_model,
        validate_autoencoder_fn=validate_model,
        plot_losses_fn=plot_losses,
        loss_fn=loss_fn,
        lr=Conf.mutation_lr,
        weight_decay=Conf.mutation_weight_decay,
        training_graph_path="results/mutation/cv/mutation_autoencoder_cv",
        latent_dim=Conf.latent_dim,
        input_size=input_size,
        n_splits=5,
        num_epochs=Conf.num_epochs,
        batch_size=Conf.batch_size,
        shuffle_dataset=True,
        random_state=int(os.environ['PROJECT_SEED'])
    )

    for fold_info in fold_results:
        print(f"Fold {fold_info['fold']} results:")
        print(f"  Best Epoch: {fold_info['best_epoch']}")
        print(f"  Best Validation Loss: {fold_info['best_val_loss']:.4f}")
        print("-" * 40)