# Efficient Semantic Segmentation

## Architecture description

The project is based on the the [U-Net](https://arxiv.org/abs/1505.04597) architecture and uses ideas of the [MobileNetV2](https://arxiv.org/abs/1801.04381), especially  the encoder which is through the depthwise convolution about 10 times more efficient than with a normal convolution. The architecture consists of two main components:

1. The `inverted residual block`

it the main building block of the encoder.  
It expands the channels of the input by a fixed factor (often 4-6) via a convolution with a 1x1 filter. 
Applies a depthwise convolution with a 3x3 filter to create an independent filter for every channel present.
And in the end reduced the channels via another 1x1 convolution down to the output channel size.

After every convolution batch normalization is applied and after the first two an additionall linear activation, the ReLU6, which is a normal ReLU that has an upper limit of 6.

2. The `U-Net`

consists of an encoder that reduces the spatial dimensions (width and height) of the input and a decoder that increases the spatial dimensions again.  
The encoder is based on the MobileNetV2 architecture and consists of a sequence of inverted residual blocks while the decoder consists of a sequence of upsampling blocks (bilinear) with skip connections from the encoder.

In my case the **encoder** consists of 5 layers. One initial convolution layer to expand the input channels from 3 RGB channels to 16 channels and then 4 `inverted residual blocks` with the following parameters:

| Block | Expansion factor | Output channels | Stride      | Repeats |
|-------|------------------|-----------------|-------------|---------|
| 1     | 4 (can be 1)     | 32              | 2 (can be 1)| 1       |
| 2     | 4                | 64              | 2           | 2       |
| 3     | 4                | 128             | 2           | 3       |
| 4     | 4                | 256             | 2           | 1       |

The smaller the spatial dimensions the more channels are present and the more often the `inverted residual block` is repeated. 

The **decoder** consists of 4 upsampling blocks and a final convolutional layer to reduce the output channels to the number of classes, in this case 19. The upsampling block consists of four steps:

1. Bilinear upsampling to double the spatial dimensions
2. Concatenation with the skip connection from the encoder at the same spatial dimensions
3. Convolution with a 1x1 filter to reduce the number of channels
4. Refine with a 3x3 (depthwise) convolution (may be repeated)

The parameters of the upsampling blocks are as follows:
| Block | Output channels | Skip connection from encoder |
|-------|-----------------|------------------------------|
| 1     | 128             | 4                            |
| 2     | 64              | 3                            |
| 3     | 32              | 2                            |
| 4     | 16              | 1                            |


**OPEN QUESTIONS**: 
- How many times should the 3x3 convolution be repeated?
- What how efficient is it to use a depthwise convolution in the decoder?

## Training setup
The model is trained on the Cityscapes dataset, which consists of 5000 images with size 1024x2048 pixels of urban street scenes with pixel-wise annotations for 30 classes of which 19 are considered in this work. The dataset is split into a training set of 2975 images, a validation set of 500 images. (The test set would be another 1525 images but is not used in this work).

The model is trained for 30 epochs with a batch size of 6 and a learning rate of 0.001. I used the cross-entropy to calculate the loss and an AdamW optimizer to update the weights. Additionally, I used a learning rate scheduler (reduce on plateau) to reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 epochs.

While training I'm calculating the mIoU (mean Intersection over Union) as well as the pixel accuracy on the validation set after every epoch to monitor the performance of the model. In addition, I'm generating an image with the predicted segmentation mask to evaluate the performance.

## Results table

The best loss on the validation set is 0.345 and the best mIoU is 0.982 while this value is only achievable because the MeanIoU function has no option to ignore a specific class/label and so it also counts the ignored classes. When using the `include_background=False` parameter, the mIoU is at 0.41 but in this case the street class which is about 30%-40% of every image is also seen as background and not counted in the mIoU value. The best pixel accuracy is 0.89.

**find more information in the `report.pdf`**
