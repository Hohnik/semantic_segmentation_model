from torch import nn
import torch


class MobileNetV2Decoder(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        channels = [int(c * width_mult) for c in [256, 128, 64, 32, 16]]
        self.stage1 = DecoderBlock(channels[0], channels[1])  # 1/16
        self.stage2 = DecoderBlock(channels[1], channels[2])  # 1/8
        self.stage3 = DecoderBlock(channels[2], channels[3])  # 1/4
        self.stage4 = DecoderBlock(channels[3], channels[4])  # 1/2

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)  # 1/1
        self.final_conv = nn.Conv2d(
            channels[-1], 19, 1, bias=True
        )  # bias added like in the paper

    def forward(self, skips):
        bottleneck = skips[-1]

        x = self.stage1(bottleneck, skips[-2])
        x = self.stage2(x, skips[-3])
        x = self.stage3(x, skips[-4])
        x = self.stage4(x, skips[-5])

        x = self.upsample(x)  # 16, 256, 512
        logits = self.final_conv(x)  # 19, 256, 512
        return logits


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # TODO: concatinate (what is better addition or concatination?)

        self.compress = nn.Conv2d(in_ch + out_ch, out_ch, 1, bias=False)
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
