import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(scale, scale, ceil_mode=True)

        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.attention1(x)
        return x

class ConstBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')

        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.attention1(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels, 
            previous_channels, #list of previous output channels
            max_down_scale, #maximum down scaling to be done
            num_concat_blocks, #no. of blocks to be concatenated
            output_index, #index of output channels for each block
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()

        scale = max_down_scale 
        blocks = []
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.num_concat_blocks = num_concat_blocks
        self.UpChannel = out_channels[output_index]*self.num_concat_blocks
      

        #pos = len(in_channels) - scale - 1

        prev_ind = 1 #used for calculating index of input channels for UpBlock 

        #creating blocks with respect to the scale
        for i in range(self.num_concat_blocks):
          if scale > 0:
            block = DownBlock(in_channels[i], out_channels[output_index], pow(2,scale), **kwargs)
          elif scale == 0:
            block = ConstBlock(in_channels[i], out_channels[output_index],0, **kwargs)
          else:
            block = UpBlock(previous_channels[output_index-prev_ind], out_channels[output_index], pow(2, abs(scale)), **kwargs)
            prev_ind +=1

          blocks.append(block)
          scale = scale - 1
        
        self.blocks = nn.ModuleList(blocks)

        #concatenation block
        self.cat_block = md.Conv2dReLU(
            self.UpChannel,
            self.UpChannel,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention_cat = md.Attention(attention_type, in_channels=self.UpChannel)

    def forward(self, feature):
        
        result_list = []
        for i, block in enumerate(self.blocks):
          result = block(feature[i])
          result_list.append(result)

        concat_tensor = torch.cat(result_list, 1) #concatenation of every tensor
        final = self.cat_block(concat_tensor)
        final = self.attention_cat(final)

        return final


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class Unet3PlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        """if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )"""

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = encoder_channels[1:]
        in_channels = in_channels[::-1] #reversing the list

        out_channels = [head_channels] + list(decoder_channels) 
        previous_channels = [i * n_blocks for i in out_channels] #calculating previos channels i.e concatenated output channels of previous block
        previous_channels[0] = int(previous_channels[0] / n_blocks) #first tensor as it comes from CenterBlock, hence no concatenation

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        stages = []
        self.scale = n_blocks - 2

        #Blocks for every stage is generated
        for i in range(n_blocks):
          stage = DecoderBlock(in_channels, out_channels, previous_channels, self.scale-i, n_blocks, i + 1, **kwargs)
          stages.append(stage)
  
        self.stages = nn.ModuleList(stages)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        current_features = features[1:]
        current_features = current_features[::-1]
        current_features = list(current_features)
        
        x = self.center(head)  #first tensor is input into centr block

        decoded_features = []
        decoded_features.append(x)
        
        #tensors are calculated for each stage
        for i, stage in enumerate(self.stages):
      
          total_features = current_features.copy()
          total_features.extend(decoded_features) 
          x = stage(total_features)
          
          current_features = current_features[:-1]
          
          decoded_features = [x] + decoded_features

        return x
    
class Unet3Plus(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("Unet3Plus is not support encoder_name={}".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = Unet3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[encoder_depth - 1] * encoder_depth,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "unet3plus-{}".format(encoder_name)
        self.initialize()