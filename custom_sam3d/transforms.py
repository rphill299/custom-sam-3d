import torch
from monai.transforms import (
    Invertd,
    Activationsd,
    AsDiscreted,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CenterSpatialCropd,
)

from custom_sam3d.utils import ConvertToMultiChannelBasedOnBratsClassesd

class BratsTransforms:
    """
    Encapsulates all BRATS transforms:
    - train
    - validation
    - testing (optional)
    - post transforms (for inference)
    """

    def __init__(self, roi_size=[224, 224, 144], pixdim=(1.0, 1.0, 1.0)):
        self.pixdim = pixdim
        self.roi_size = roi_size

    def train(self) :
        return Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=self.pixdim,
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=self.roi_size, random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    def val(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=self.pixdim,
                    mode=("bilinear", "nearest"),
                ),
                # TODO: center or rand ?
                CenterSpatialCropd(keys=["image", "label"], roi_size=self.roi_size),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    def test(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=self.pixdim, mode="bilinear"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    def post(self, pre_transform):
        return Compose(
            [
                Invertd(
                    keys="pred",
                    transform=pre_transform,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                    device="cpu",
                ),
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
            ]
        )