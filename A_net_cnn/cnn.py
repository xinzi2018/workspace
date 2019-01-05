import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.temp = 1

        self.conv1 = nn.Sequential(  # (3,64,64)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,
                      stride=1, padding=2),  # (16,64,64)
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1),  # (32,32,32)
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),  # 64*32*32
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # (128,16,16)
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1),  # (128,8,8)
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),  # (256,4,4)
            nn.ReLU(),
        )

        self.linear = nn.Linear(256*4*4, 1024)
        linear_n = []
        self.value = [3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3]
        for i in self.value:
            linear_n += [nn.Linear(1024, i)]
        self.linears = nn.Sequential(*linear_n)


    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        for i in range(18):
            output.append(self.linears._modules[str(i)](x))
        return output