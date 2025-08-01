import torch
import torch.nn as nn

# Define Slice Encoder
# Expects input shape [N, C, H, W] = [N, 4, 224, 224]
class SliceEncoder(nn.Module) :
    def __init__(self, in_channels=4, kernel_size=5) :
        super(SliceEncoder, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channels, 32, self.kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, 0) # [112, 112]

        self.conv2 = nn.Conv2d(32, 64, self.kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, 0) # [56, 56]

        self.conv3 = nn.Conv2d(64, 128, self.kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, 0) # [28, 28]

        self.conv4 = nn.Conv2d(128, 256, self.kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        # FEATURE ENCODER --> NO FINAL LINEAR LAYER

    def forward(self, x, N) :
        D = x.shape[0]//N
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        skip_0 = x.reshape(N, D, C, H, W).transpose(1,2) # [N, C, D, H, W]

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        skip_1 = x.reshape(N, D, 32, H//2, W//2).transpose(1,2)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        skip_2 = x.reshape(N, D, 64, H//4, W//4).transpose(1,2)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        skip_3 = x.reshape(N, D, 128, H//8, W//8).transpose(1,2)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        return x, [skip_0, skip_1, skip_2, skip_3]

# Define volume decoder
# Expects input shape [N, C, D, H, W] = [N, 512, 144, 28, 28]
class VolumeDecoder(nn.Module) :
    def __init__(self, out_channels=3, kernel_size=5) :
        super(VolumeDecoder, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv0 = nn.Conv3d(256, 128, kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm0 = nn.InstanceNorm3d(128)
        self.relu0 = nn.LeakyReLU()

        self.conv1 = nn.Conv3d(256, 64, kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm3d(64)
        self.relu1 = nn.LeakyReLU()
        self.usmp1 = nn.Upsample([144, 56, 56])

        self.conv2 = nn.Conv3d(128, 32, kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm3d(32)
        self.relu2 = nn.LeakyReLU()
        self.usmp2 = nn.Upsample([144, 112, 112])

        self.conv3 = nn.Conv3d(64, 16, kernel_size, 1, 'same', bias=False, padding_mode='reflect')
        self.norm3 = nn.InstanceNorm3d(16)
        self.relu3 = nn.LeakyReLU()
        self.usmp3 = nn.Upsample([144, 224, 224])

        self.final_conv = nn.Conv3d(20, self.out_channels, kernel_size, 1, 'same', bias=True, padding_mode='reflect')

    def forward(self, x, skip_connections) :
        skip_0, skip_1, skip_2, skip_3 = skip_connections

        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)

        x = torch.cat((x,skip_3), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.usmp1(x)

        x = torch.cat((x, skip_2), 1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.usmp2(x)

        x = torch.cat((x, skip_1), 1)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.usmp3(x)

        x = torch.cat((x, skip_0), 1)

        x = self.final_conv(x)

        return x

# Define custom SAM3D model
class CustomSAM3D(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size=5) :
        super(CustomSAM3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        #2D Encoder to use
        self.slice_encoder = SliceEncoder(in_channels, kernel_size)

        #3D Decoder to use
        self.volume_decoder = VolumeDecoder(out_channels, kernel_size)

    def forward(self, x) :
        #get input shapes
        N = x.shape[0]
        C = x.shape[1]
        D = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]
        # [N, C, D, H, W]

        # reshape for slice encoder
        x = x.transpose(1,2) # [N, D, C, H, W]
        x = x.reshape(-1, C, H, W) # [N*D, C, H, W]

        # pass to slice encoder
        x, skip_connections = self.slice_encoder(x, N) # [N*D, 256, H/8, W/8]

        # reshape for volume decoder
        newC, newH, newW = x.shape[1], x.shape[2], x.shape[3]

        x = x.reshape(N, D, newC, newH, newW) # [N, D, 256, H/8, W/8]
        x = x.transpose(1,2) # [N, 256, D, H/8, W/8]

        # pass to volume decoder
        x = self.volume_decoder(x, skip_connections) #[N, 3, D, H, W]
        return x