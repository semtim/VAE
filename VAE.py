import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from plasticc_gp import plasticc_gp
import numpy as np
import matplotlib.pyplot as plt
import random

# Уменьшаем вариативность весов
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Загрузка данных
data, metadata = plasticc_gp()
data = data[:, :-11]
filtered_data = data[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]

# Определение VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 7),
            nn.ReLU(),
            nn.Linear(7, 2),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(2, latent_dim)
        self.logvar_layer = nn.Linear(2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 7), 
            nn.ReLU(),
            nn.Linear(7, input_dim)
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
def loss_function(recon_x, x, mu, logvar, kld_weight=1.5, recon_weight=1.0):
    BCE = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = kld_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Подготовка данных
class MyDataset(Dataset):
    def __init__(self, filtered_data):
        self.data = filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Разделение данных на обучающую и валидационную выборки
train_data = filtered_data[:int(0.7 * len(data))]
val_data = filtered_data[int(0.3 * len(data)):]

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=22, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
val_dataloader = DataLoader(val_dataset, batch_size=22, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))
#train_dataloader = DataLoader(train_dataset, batch_size=22, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=22, shuffle=False)

# Параметры модели
input_dim = data.shape[1]
latent_dim = 2

# Создание модели
model = VariationalAutoencoder(input_dim, latent_dim)

# Оптимизация
optimizer = optim.SGD(model.parameters(), lr=0.000025)

# Обучение модели
epochs = 50
kld_weight = 1.5
recon_weight = 1.0

train_losses = []
val_losses = []

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
    
    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")

    # Оценка на валидационной выборке
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch: {epoch+1}, Validation Loss: {val_loss}")

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Поиск аномалий
threshold = 0.1  # Порог для обнаружения аномалий
anomalies = []
with torch.no_grad():
    for i, sample in enumerate(data):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        output, mu, logvar = model(sample_tensor)
        reconstruction_error = nn.functional.mse_loss(output, sample_tensor).item()
        if reconstruction_error > threshold:
            anomalies.append(i)

print("Аномалии найдены в следующих индексах:", anomalies)
print("Количество аномалий:", len(anomalies))

# Визуализация распределения латентных векторов
latent_vectors = []

model.eval()
with torch.no_grad():
    for batch in DataLoader(MyDataset(data), batch_size=32, shuffle=False):
        mu, logvar = model.encode(batch)
        z = model.reparameterize(mu, logvar)
        latent_vectors.append(z)

latent_vectors = torch.cat(latent_vectors, dim=0).numpy()

# Убираем -inf и inf из латентных векторов
latent_vectors = np.nan_to_num(latent_vectors, nan=0.0, posinf=0.0, neginf=0.0)

for i in range(latent_dim):
    plt.figure(figsize=(8, 4))
    plt.hist(latent_vectors[:, i], bins=30, density=True, alpha=0.6, color='g')
    plt.title(f'Гистограмма для латентного пространства {i+1}')
    plt.xlabel('Значение')
    plt.ylabel('Плотность')
    plt.grid(True)
    plt.show()
 