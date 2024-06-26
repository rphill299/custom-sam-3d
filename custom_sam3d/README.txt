Code Breakdown:
	- All custom code was written by Ryan P Hill.  Please reach out to rphill@umass.edu if any problems occur.

	- All external code came from this MONAI tutorial: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

	- brats_segmentation_3d_original is my adaptation of the above tutorial to work on my device/colab.  Mostly everything in my file is the same as the original tutorial, except very small changes (i.e. the directory for saving/loading models, the number of threads to fit the number of cores available, the number of epochs, enable training on CPU, etc.)

	- brats_segmentation_3d_custom_SAM3D.2 showcases the bulk of my work.  The notable additions are (line numbers include outputs and are approximate):
		- (lines 142-146) added a code block to reload dataset from tmp directory, instead of downloading it again within the same runtime.
		- (line 219) changed the val_transform to crop the validation data to the same size as the train_transform
			- original model could somehow properly handle different input crop sizes; our model expects 4x224x224x144 volumes.
		- (lines 277-408) created new custom model Custom_SAM3D_RPH, following a U-Net and SAM3D architecture, featuring
			- custom 2d slice encoder (down convolutions)
			- custom 3d slice decoder (up convolutions)
		- (lines 427-431) loading previous best model to resume training from a previous epoch
			- frequent crashes meant iterative training
		- (line 448) changed crop size for sliding_window_inference to fit model expectations.
		- (lines 498, 503, 510, 537, 539) permuting inputs/outputs to fit expected indexing order of PyTorch vs. MONAI

Instructions for Inference:
	- Run the inference notebook in Colab. (May require GPU)

Instructions for Training:
	- Run one of the brats_segmentation_3d notebooks in Colab. (May require GPU)
		- Takes 20 minutes per epoch on my machine using colab's T4.