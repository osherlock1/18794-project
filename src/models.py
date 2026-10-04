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
