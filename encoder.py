from torch import nn


class MobileNetV2Encoder(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0):
        super().__init__()

        channels = [int(c * width_mult) for c in [16, 32, 64, 128, 256]]
        blocks = [int(b * depth_mult) for b in [1, 2, 3, 1]]

        self.first_layer = nn.Conv2d(
            in_channels=3,
            out_channels=channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )  # img size 1/2
        self.first_norm = nn.BatchNorm2d(channels[0])
        self.first_relu = nn.ReLU6()

        self.stage1 = self.create_block(channels[0], channels[1], blocks[0], 4)  # 1/4
        self.stage2 = self.create_block(channels[1], channels[2], blocks[1], 4)  # 1/8
        self.stage3 = self.create_block(channels[2], channels[3], blocks[2], 4)  # 1/16
        self.stage4 = self.create_block(channels[3], channels[4], blocks[3], 1)  # 1/32

    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_norm(x)
        s0 = self.first_relu(x)

        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        return [s0, s1, s2, s3, s4]

    def create_block(self, in_ch: int, out_ch: int, blocks: int, expansion: int):
        layers = nn.Sequential()
        layers.append(InvertedResidualBlock(in_ch, out_ch, expansion, stride=2))
        for _ in range(blocks - 1):
            layers.append(InvertedResidualBlock(out_ch, out_ch, expansion, stride=1))

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
            expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False
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
