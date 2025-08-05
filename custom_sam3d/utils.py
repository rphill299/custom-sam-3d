from monai.transforms import MapTransform
import torch

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert labels to multi channels based on brats classes:
        label 1 is the peritumoral edema
        label 2 is the GD-enhancing tumor
        label 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge label 2 and label 3 to construct TC
                result.append(torch.logical_or(d[key] == 2, d[key] == 3))
                # merge labels 1, 2 and 3 to construct WT
                result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
                # label 2 is ET
                result.append(d[key] == 2)
                d[key] = torch.stack(result, axis=0).float()
            return d