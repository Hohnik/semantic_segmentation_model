from torch import nn
import torch


class MobileNetV2Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DecoderBlock(320, 96, 64)  # 1/16
        self.block2 = DecoderBlock(64, 32, 48)  # 1/8
        self.block3 = DecoderBlock(48, 24, 32)  # 1/4
        self.block4 = DecoderBlock(32, 16, 16)  # 1/2

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)  # 1/1
        self.final_conv = nn.Conv2d(
            16, 19, 1, bias=True
        )  # bias added like in the paper

    def forward(self, skips):
        bottleneck = skips[6]  # s7

        x = self.block1(bottleneck, skips[4])  # s5
        x = self.block2(x, skips[2])  # s3
        x = self.block3(x, skips[1])  # s2
        x = self.block4(x, skips[0])  # s1

        x = self.upsample(x)  # 16, 256, 512
        logits = self.final_conv(x)  # 19, 256, 512
        return logits


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # TODO: concatinate (what is better addition or concatination?)

        self.compress = nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False)
        self.refine1 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False
        )
        self.refine2 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.concat([x, skip], dim=1)  # dim 1 -> channels (B, C, H, W)
        x = self.compress(x)
        x = self.refine1(x)
        x = self.refine2(x)
        return x
