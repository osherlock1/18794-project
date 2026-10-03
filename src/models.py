from torch import nn


class CNNBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1086, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 12, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=250),
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.features(x)
        x = self.classifier(x)
        return x
