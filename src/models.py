import torch
from torch import nn


class CNNBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1086, out_channels=192, kernel_size=17, stride=1, padding=8
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(
                in_channels=192, out_channels=192, kernel_size=17, stride=1, padding=8
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(
                in_channels=192, out_channels=192, kernel_size=17, padding=8, stride=1
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=192 * 32, out_features=256 * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(in_features=256 * 2, out_features=250),
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=150, out_channels=192, kernel_size=5, stride=1, padding=8
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(
                in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=8
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(
                in_channels=192, out_channels=192, kernel_size=5, padding=8, stride=1
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2),
        )


class CNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedder = CNNEmbedder()

        self.position = nn.Parameter(torch.randn(1, 50, 192) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=4,
            dim_feedforward=384,
            dropout=0.2,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(192, 250),
        )

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.embedder.features(x)
        x = x.transpose(1, 2)

        time_steps = x.size(1)
        if time_steps > self.position.size(1):
            raise ValueError("Sequence exceeds max_toxens")

        x = x + self.position[:, :time_steps]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
