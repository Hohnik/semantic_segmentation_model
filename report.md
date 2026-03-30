# Efficient Semantic Segmentation

## Architecture description

The project is based on the the [U-Net](https://arxiv.org/abs/1505.04597) architecture and uses ideas of the [MobileNetV2](https://arxiv.org/abs/1801.04381), especially  the encoder.

1. The `inverted residual block`

This block expands the channels of the input by a fixed factor (often 4-6) via a convolution with a 1x1 filter. 
Applies a depthwise convolution with a 3x3 filter to create an independent filter for every channel present.
In the end the channels are reduced via another 1x1 convolution down to the output channel size.

After every convolution a batch normalization is applied and after the first two an additionall linear activation, the ReLU6, which is a normal ReLU that has an upper limit of 6.



## Training setup


## Results table


## Qualitative examples


## Answers to the questions below


