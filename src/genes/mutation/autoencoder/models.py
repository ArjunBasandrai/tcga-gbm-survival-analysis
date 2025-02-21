from torch import nn

class ClinicalAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(ClinicalAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class MutationAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(MutationAE, self).__init__()

        self.embedding = nn.Embedding(input_size, 2048)
        
        self.encoder = nn.Sequential(
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x.long())
        embedded = embedded.mean(dim=1)
        latent = self.encoder(embedded)
        reconstructed = self.decoder(latent)
        return reconstructed, latent