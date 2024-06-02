# Introduction {#sec:intro}

Brain tumor segmentation technologies are used by doctors to diagnose
patients with brain tumors. This involves first taking an MRI image of
the patient's brain, which results in a 3D volume, then feeding this 3D
volume to some model that makes voxel-level predictions if a part of the
brain is a tumor or not.

Figure
[\[fig:inputs-outputs\]](#fig:inputs-outputs){reference-type="ref"
reference="fig:inputs-outputs"} shows a 2D slice of our input and output
volumes. The input consists of four channels corresponding to the four
MRI modalities used, and the output consists of three channels
corresponding to the three types of tumor tissue we want prediction
masks for. MONAI uses this multi-channel approach to multi-class
segmentation, as opposed to outputting a single non-binary channel. This
is useful, as it boils down the problem to transforming a four channel
volume to a three channel volume of the same spatial dimension. It also
is possible, and common, for a part of the brain to fall into multiple
categories, so it makes sense to predict each output category separately
in a binary fashion, instead of a predicting a single label per-voxel.

![image](latex/mri.png){width="\\linewidth"}
![image](latex/tumor_prediction.png){width="\\linewidth"}

# Related Work {#sec:related-work}

Bui, et al. [@bui2024sam3d] did similar work on SAM3D related to brain
tumor segmentation, but they were more generally focused on segmentation
of any medical imaging (Figure [1](#fig:sam3d_bui){reference-type="ref"
reference="fig:sam3d_bui"}). They also included experiments for organ
segmentation, lung tumor segmentation, and automatic cardiac diagnosis,
and these datasets all feature single channel input volumes.

This is the major difference in their approach from ours. Since they had
more single channel input volumes to consider, they used grey scale
image slices as inputs to their image decoder, while we use all four
channels of the brain MRI. Since their image decoder expected RGB images
with three channels, the authors tiled the 1xHxW grey scale inputs to be
3xHxW, where each channel is the same. We flagged this as a flaw to fix,
as the spatial connections between channels are pointless if each
channel is the same.

They did not specify how they handled multi-channel inputs, although
they do mention that their model for brain tumor segmentation featured
4x the number of parameters as the other models, so it's possible they
process each channel separately through its own network and perform some
kind of pooling. It's better to make predictions by considering
connections across channels, rather than doing a final pooling at the
end, so we process our slices as four channel images.

![Original SAM3D network proposed by Bui, et al. [@bui2024sam3d]. The
major difference between this network and our network is the handling of
multi-channel inputs. We keep the multiple channels together, while they
process each channel individually as tiled greyscale (1xHxW $\to$
3xHxW).](latex/sam3d_bui.png){#fig:sam3d_bui width="\\linewidth"}

# Method {#sec:method}

Our overall model architecture follows a typical U-Net structure (Figure
[2](#fig:model-architecture){reference-type="ref"
reference="fig:model-architecture"}). We use a slice encoder to decrease
spatial dimension while increasing feature dimension, then use a volume
decoder to increase spatial dimension back to original while decreasing
feature dimension. We have skip connections connecting volumes of the
same spatial dimension from the encoder to decoder, and a final
convolution layer produces three channels for our three predicted masks.

We start by splitting our input volume into slices, then passing each to
a 2D encoder before stacking the encoding into a volume again and
feeding that to a 3D decoder.

## Slice Encoder

Our slice encoder is based upon the PyTorch implementation
[@Ansel_PyTorch_2_Faster_2024] of DenseNet [@huang2018densely], which
features an initial Conv2d-BatchNorm2d-ReLU-MaxPool2d block before the
dense convolution layers. We repeat this initial block structure three
times, reducing the image spatial dimension by eight. To store skip
connections, each intermediate group of slices is grouped into a volume
and forwarded to the decoder as a volume of decoded slices.

We initially planned on using a vision transformer for encoding, but
even the smallest model didn't fit in our RAM with a batch size of one.

## Volume Decoder

Our volume decoder is based upon the volume decoder of SAM3D
[@bui2024sam3d], which uses four
Conv3d-InstanceNorm3d-LeakyReLU-Upsample blocks before a final
prediction block. We simply use three of these blocks so our upsampling
matches our downsampling. Since our downsampling takes place in the 2D
context, our 3D upsampling only takes place along the HxW dimensions
(depth dimension is untouched).

We finish by passing our final upsampled features to another Conv3d
layer for predicting our three output masks.

![Our Custom SAM3D
architecture.](latex/model_arch.png){#fig:model-architecture
width="\\linewidth"}

## Error Metric

We use the standard DICE error metric of two volumes, which measures
$$\frac{2 \times \text{intersection}}{\text{union}}$$ and ranges from
zero (no overlap) to one (perfect overlap). We compute the DICE metric
for each output channel, then average these results for our average DICE
metric, which we use to compare two models overall.

# Experiments {#sec:experiments}

## MONAI

Much of our training pipeline was borrowed from a Google Colab tutorial
from MONAI [@monai]. MONAI, or Medical Open Network for AI, is an open
source framework dedicated to loading medical datasets, designing
models, and running training and inference pipelines (Figure
[3](#fig:monai){reference-type="ref" reference="fig:monai"}). The MONAI
framework is built entirely on top of PyTorch, so it's relatively
straight-forward to incorporate a custom PyTorch model into the
workflow.

![Typical MONAI training pipeline. Our focus was on changing the
model.](latex/monai.png){#fig:monai width="\\linewidth"}

## Dataset

We start by loading the BraTS dataset from Medical Decathlon, a common
source for loading medical imaging data, which is easily loaded by
MONAI. The BraTS dataset is used each year for the annual BraTS Tumor
Segmentation challenge, and Medical Decathlon gives us easy access to
the 2016 and 2017 BraTS datasets. This includes 750 four channel 3D
volumes (484 training, 266 testing) of brain MRI images. Each volume has
been pre-processed to remove the skull, so we're left with simply the
brain.

The MONAI framework already loads the datasets for us, and uses
different \"transforms\" which are supposed to enable dynamic data
augmentation by randomly adjusting intensities, though this is something
we could not confirm, lacking extensive knowledge of MONAI.

We used a batch size of one, as some of our original experiments didn't
support larger batch sizes. Exploring adaptive batch sizing may change
training or inference behavior. We keep the batch size of one, as our
training data is so small, so we want to maximally learn from each
training example.

## Training

First, we ran the original MONAI tutorial, with the given network they
defined. The tutorial defined a SegResNet based upon [@myronenko20183d],
which uses residual connections and a variational autoencoder to make
tumor mask predictions. The original SegResNet model has great final
results, achieving over a 0.9 average DICE metric, and the MONAI
implementation achieves about 0.8 average DICE metric after training, so
this model is a good baseline to compare our model.

We ran the MONAI model for ten epochs and achieved an average validation
DICE metric of 0.675.

Next, we ran our custom network for ten epochs, and achieved an average
DICE metric of 0.7286.

## Evaluation

Below shows the best metrics achieved by each model in ten epochs, as
well as mask predictions made by our Custom SAM3D on a given input.

              SegResNet   Our Custom SAM3D
  ----------- ----------- ------------------
  Epochs                  
  Mean DICE               **0.7286**
  TC DICE                 **0.7794**
  WT DICE                 **0.8811**
  ET DICE                 **0.5253**

![image](latex/pred_input.png){width="\\linewidth"}
![image](latex/pred_label.png){width="\\linewidth"}
![image](latex/pred_output.png){width="\\linewidth"}

# Conclusion and Further Work {#sec:conc}

This paper lays out a great blueprint for future experiments to build
upon. Even though we're using a relatively shallow custom model, with a
few easily-identifiable improvements needed, we still perform better
than the pre-built MONAI SegResNet model for the first ten epochs. The
SegResNet model is based on a paper that achieves over a 0.9 DICE
metric, which is considered nearly state-of-the-art, so the hope is our
model architecture would show similar results if expanded and trained
longer.

Expanding the network would mean simply making it deeper. Adding more
convolution layers, or adding more channels to layer, would make the
model deeper and potentially improve results. Specifically, it should
help to add more layers in the decoder when processing a new skip
connection. This is typically done in a few layers per skip connection,
while we only use one layer to reduce model size. We should also add the
original input volume as a skip connection for our final prediction
layer, as we originally overlooked this skip connection.

The task of 3D volume segmentation is computationally expensive, so we
cannot speak conclusively on this model until hundreds of epochs have
been run across many many hours, as is typically done for this task. But
we can say that we show strong promise, as our early performance closely
matches that of a nearly state-of-the-art model. More exploration is
needed to determine the full capabilities (and limitations) of 3D
segmentation via 2D slice segmentation, however, we've shown that we
should expect good results if we continue to pursue this method, and
potentially even a new state-of-the-art.
