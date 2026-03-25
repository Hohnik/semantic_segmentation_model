from torch import nn


class MobileNetV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Conv2d(
            3, 32, 3, stride=2, padding=1, bias=False
        )  # img size: 1/2
        self.first_norm = nn.BatchNorm2d(32)
        self.first_relu = nn.ReLU6()

        self.block1 = self.create_block(32, 16, 1, 1, 1)  # 1/2
        self.block2 = self.create_block(16, 24, 6, 2, 2)  # 1/4
        self.block3 = self.create_block(24, 32, 6, 2, 3)  # 1/8
        self.block4 = self.create_block(32, 64, 6, 2, 4)  # 1/16
        self.block5 = self.create_block(64, 96, 6, 1, 3)  # 1/16
        self.block6 = self.create_block(96, 160, 6, 2, 3)  # 1/32
        self.block7 = self.create_block(160, 320, 6, 1, 1)  # 1/32

    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_norm(x)
        x = self.first_relu(x)

        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        s5 = self.block5(s4)
        s6 = self.block6(s5)
        s7 = self.block7(s6)  # bottleneck

        return s1, s2, s3, s4, s5, s6, s7

    def create_block(
        self, in_ch: int, out_ch: int, expansion: int, stride: int, blocks: int
    ):
        layers = nn.Sequential()
        layers.append(InvertedResidualBlock(in_ch, out_ch, expansion, stride))

        for _ in range(blocks - 1):
            layers.append(InvertedResidualBlock(out_ch, out_ch, expansion, 1))

        return layers


class InvertedResidualBlock(nn.Module):  # Secret souce
    def __init__(
        self, in_channels: int, out_channels: int, expansion_factor: int, stride: int
    ):
        super().__init__()
        expanded = in_channels * expansion_factor

        self.expand = nn.Conv2d(
            in_channels, expanded, 1, bias=False
        )  # 32 -> 160 channels with 160 filters 1x1x32 (5120 parameters)
        self.first_norm = nn.BatchNorm2d(expanded)
        self.expand_relu = nn.ReLU6()

        self.conv = nn.Conv2d(
            expanded, expanded, 3, stride, padding=1, groups=expanded, bias=False
        )  # 160 filters with 3x3x1 (1440 parameters) and shrinks the images size
        # groups makes it so that each channel has its own filter
        self.second_norm = nn.BatchNorm2d(expanded)
        self.conv_relu = nn.ReLU6()

        self.shrink = nn.Conv2d(
            expanded, out_channels, 1, bias=False
        )  # 160 -> 16 with 16 filters 1x1x160 (2560 parameters)
        self.third_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.expand(x)
        out = self.first_norm(out)
        out = self.expand_relu(out)

        out = self.conv(out)
        out = self.second_norm(out)
        out = self.conv_relu(out)

        out = self.shrink(out)
        out = self.third_norm(out)

        return out
