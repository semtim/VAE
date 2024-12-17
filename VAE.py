import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from plasticc_gp import plasticc_gp
import numpy as np

# Загрузка данных
data, metadata = plasticc_gp()

class GaussianActivation(nn.Module):
    def __init__(self):
        super(GaussianActivation, self).__init__()

    def forward(self, x):
        return torch.exp(-x ** 2)

# Определение VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            GaussianActivation(),
            nn.Linear(4, 2),
            GaussianActivation()
        )
        self.mu_layer = nn.Linear(2, latent_dim)
        self.logvar_layer = nn.Linear(2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2),
            GaussianActivation(),
            nn.Linear(2, 4), 
            GaussianActivation(),
            nn.Linear(4, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Функция потерь для VAE
def loss_function(recon_x, x, mu, logvar, kld_weight=0.005, recon_weight=1.0):
    BCE = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = kld_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Подготовка данных
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Разделение данных на обучающую и валидационную выборки
train_data = data[:int(0.7 * len(data))]
val_data = data[int(0.3 * len(data)):]

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)

# Параметры модели
input_dim = data.shape[1]
latent_dim = 2
# Создание модели
model = VariationalAutoencoder(input_dim, latent_dim)

# Оптимизация
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
epochs = 50
kld_weight = 0.005
recon_weight = 1.0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch: {epoch+1}, Loss: {train_loss / len(train_dataloader.dataset)}")

# Оценка модели
model.eval()
val_loss = 0
with torch.no_grad():
    for batch in val_dataloader:
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
        val_loss += loss.item()

print(f"Validation Loss: {val_loss / len(val_dataloader.dataset)}")

# Поиск аномалий
threshold = 0.2  # Установите порог для обнаружения аномалий
anomalies = []
with torch.no_grad():
    for i, sample in enumerate(data):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        output, mu, logvar = model(sample_tensor)
        reconstruction_error = nn.functional.mse_loss(output, sample_tensor).item()
        if reconstruction_error > threshold:
            anomalies.append(i)

print("Аномалии найдены в следующих индексах:", anomalies)