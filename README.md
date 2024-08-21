# C-SAM3D

[[Paper]](https://github.com/rphill299/custom-sam-3d/blob/main/project/paper_draft.pdf)
[[SAM3D Paper]](https://arxiv.org/html/2309.03493v4)
[[BraTS Challenge]](https://www.med.upenn.edu/cbica/brats/)
[[MONAI Framework]](https://monai.io/)
[[MONAI Tutorial]](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)
[[Medical Decathlon Datasets]](http://medicaldecathlon.com/)


See a problem? [Open an issue](https://github.com/rphill299/custom-sam-3d/issues/new)

C-SAM3D is a model architecture developed for segmentation tasks on multi-channel 3D volumes, specifically brain tumor segmentation within MRIs.  

The C-SAM3D model is inspired by the SAM3D model, which encodes 3D medical images slice-by-slice as 2D images, then combines and decodes all slices into a 3D output volume of voxel-wise predictions.  We developed C-SAM3D within an existing brain tumor segmentation tutorial using 2016-17 BraTS data for training and testing.

### Inputs
![prediction input](/photos/pred_input.png)
### Expected Outputs
![prediction label](/photos/pred_label.png)
### C-SAM3D Predictions
![prediction output](/photos/pred_output.png)

## Approach

![C-SAM model architecture](/photos/model_arch.png)

We use a standard U-Net architecture to transform our 4-channel input volume (brain MRI) into a 3-channel output volume (brain tumor segmentation) using convolution and down/upsampling.  Each input channel corresponds to a different MRI modality.  Each output channel corresponds to a different part of the brain tumor.  A single voxel can be multiple parts of the brain tumor, so we use multi-channel binary segmentation, rather than outputting a single channel with several possible lables.

## Setup and Usage

To ensure all paths are valid, we recommend downloading our repo then uploading the "project" folder to your Google Drive.  Then open the desired .ipynb file in Google Colab for training or inference.

### Inference

inference.ipynb loads a pretrained model and performs inference on the validation and test sets.

### Training

Other .ipynb notebooks showcase the training procedure.
