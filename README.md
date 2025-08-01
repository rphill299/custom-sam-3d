# C-SAM3D

[[My Paper]](https://github.com/rphill299/custom-sam-3d/blob/main/project/paper_draft.pdf)
[[SAM3D Paper]](https://arxiv.org/html/2309.03493v4)
[[BraTS Challenge]](https://www.med.upenn.edu/cbica/brats/)
[[MONAI Framework]](https://monai.io/)
[[MONAI Tutorial]](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
[[Medical Decathlon Datasets]](http://medicaldecathlon.com/)


See a problem? [Open an issue](https://github.com/rphill299/custom-sam-3d/issues/new)

C-SAM3D is a model architecture developed for segmentation tasks on multi-channel 3D volumes, specifically brain tumor segmentation within MRIs.

The C-SAM3D model is inspired by the SAM3D model, which encodes 3D medical images slice-by-slice as 2D images, then combines and decodes all slices into a 3D output volume of voxel-wise predictions.  We developed C-SAM3D within an existing brain tumor segmentation tutorial using 2016-17 BraTS data for training and testing.

## Approach

![C-SAM model architecture](/figures/high_level_sam3d.png)

We use a standard U-Net architecture to transform our 4-channel input volume (brain MRI) into a 3-channel output volume (brain tumor segmentation) using convolution and down/upsampling.  Our encoder processes the volume in 2D while the decoder uses 3D convolution.  The model includes skip connections between similarly sized blocks.

## Inputs & Outputs
#### Inputs
Each input channel corresponds to a different MRI modality (FLAIR, T1w, T1gd, T2w).
![prediction input](/figures/pred_input.png)
#### True Outputs
Each output channel corresponds to a different part of the brain tumor (tumor core, whole tumor, enhancing tumor).
![prediction label](/figures/pred_label.png)
#### C-SAM3D Predictions
Since the classes are not mutually exclusive, we use binary segmentation across three channels for our three targets.
![prediction output](/figures/pred_output.png)


## Setup and Usage

#### Local
_coming soon_

#### Google Colab

To ensure all paths are valid, we recommend moving our entire repo to your Google Drive then opening the desired notebook in Google Colab.

## Inference

inference.ipynb loads a pretrained model and performs inference on the validation and test sets.

## Training

brats_segmentation_3d_custom_SAM3D.ipynb showcases the training procedure.
