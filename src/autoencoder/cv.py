import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import os

def k_fold_autoencoder_training(
    model_ae,
    optimizer,
    scheduler,
    early_stopping,
    device,
    dataset,
    train_autoencoder_fn,
    validate_autoencoder_fn,
    plot_losses_fn,
    loss_fn,
    lr,
    weight_decay,
    latent_dim,
    input_size,
    n_splits=5,
    num_epochs=100,
    batch_size=32,
    shuffle_dataset=True,
    random_state=42,
    print_logs=False,
    print_training_graph=True,
    training_graph_path="results/ae_training_graph",
):

    kfold = KFold(n_splits=n_splits, shuffle=shuffle_dataset, random_state=random_state)
    
    fold_results = []

    indices = list(range(len(dataset)))

    for fold, (train_ids, val_ids) in enumerate(tqdm(kfold.split(indices), total=kfold.get_n_splits())):
        if print_logs:
            print(f"\n===== Fold {fold + 1}/{n_splits} =====")
        
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_ae(input_size, latent_dim=latent_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        early_stopping.reset()
        
        train_loss_history = []
        val_loss_history = []

        for epoch in range(num_epochs):
            train_loss = train_autoencoder_fn(model, optimizer, loss_fn, train_loader, device)
            val_loss = validate_autoencoder_fn(model, loss_fn, val_loader, device)

            scheduler.step()

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if print_logs:
                print(f"Fold {fold + 1}/{n_splits}:\n" )
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train -> Loss: {train_loss:.4f}")
                print(f"  Val   -> Loss: {val_loss:.4f}")

            if early_stopping(epoch, val_loss, model):
                if print_logs:
                    print(f"\nEarly Stopping triggered at Epoch {early_stopping.best_epoch + 1} "
                          f"with Best Validation Loss: {early_stopping.best_loss:.4f}")
                break

        model.load_state_dict(torch.load(early_stopping.path, weights_only=True))

        fold_info = {
            "fold": fold + 1,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "best_epoch": early_stopping.best_epoch + 1,
            "best_val_loss": early_stopping.best_loss
        }
        fold_results.append(fold_info)
        if print_logs:
            print(f"Fold {fold + 1}/{n_splits}: Best Val Loss = {early_stopping.best_loss:.4f}")

        if print_training_graph:
            os.makedirs(os.path.dirname(training_graph_path), exist_ok=True)
            plot_losses_fn(train_loss_history, val_loss_history, f"{training_graph_path}_{fold}.png", clip=False, display=False)

    return fold_results