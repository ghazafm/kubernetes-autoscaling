import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: tensor state, shape [batch, state_dim]

        # Output dari layer ini adalah nilai yang akan digunakan di layer berikutnya
        # Contoh input: [0.5, 0.2, 0.1], bobot: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        # Hasil perkalian matriks: [0.5*0.1+0.2*0.2+0.1*0.3, 0.5*0.4+0.2*0.5+0.1*0.6]
        # = [0.5, 0.38]
        x = self.fc1(x)

        # Batch normalization for training stability
        # menormalisasi menggunakan mean dan varians
        # menggunakan output dari layer sebelumnya
        # Hasil dari batch normalization ini adalah gamma * normalized + beta
        # Contoh output sebelum bn: [1.0, 2.0, 3.0], mean=2.0, var=1.0
        # Normalized = (x - mean) / sqrt(var + epsilon)
        # Hasilnya: [-1.0, 0.0, 1.0]
        # Dengan gamma=1, beta=0, output akhir bn: [-1.0, 0.0, 1.0]
        x = self.bn1(x)

        # Relu membuat nilai negatif dari layer sebelumnya menjadi 0
        # Contoh [-0.5, 0.2, -1.0, 1.5] menjadi [0.0, 0.2, 0.0, 1.5]
        x = F.relu(x)

        # Dropout untuk regularisasi
        # Contoh: dengan p=0.2, 20% neuron akan di-set ke 0 secara acak
        x = self.dropout(x)

        # Layer kedua
        # Contoh input: [0.5, 0.2, 0.1], bobot: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        # Output layer tanpa aktivasi
        # Contoh:
        return self.fc_out(x)
