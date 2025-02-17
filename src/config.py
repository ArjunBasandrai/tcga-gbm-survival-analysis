class Conf:
    batch_size = 16
    
    latent_dim = 128

    clinical_lr = 0.001
    mutation_lr = 0.0001
    clinical_weight_decay = 0
    mutation_weight_decay = 1e-5

    num_epochs = 100