import torch
import torch.nn as nn
import torch.nn.functional as F


class binarizer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(binarizer, self).__init__()
        
        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv(1024, 512)
        self.up4 = self.double_conv(1024, 512)
        
        self.dec3 = self.upconv(512, 256)
        self.up3 = self.double_conv(512, 256)
        
        self.dec2 = self.upconv(256, 128)
        self.up2 = self.double_conv(256, 128)
        
        self.dec1 = self.upconv(128, 64)
        self.up1 = self.double_conv(128, 64)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation for binary output
    
    def double_conv(self, in_channels, out_channels):
        """Two convolutional layers followed by ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        """Upsampling using transposed convolution."""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path
        dec4 = self.dec4(bottleneck)
        up4 = self.up4(torch.cat([enc4, dec4], dim=1))
        
        dec3 = self.dec3(up4)
        up3 = self.up3(torch.cat([enc3, dec3], dim=1))
        
        dec2 = self.dec2(up3)
        up2 = self.up2(torch.cat([enc2, dec2], dim=1))
        
        dec1 = self.dec1(up2)
        up1 = self.up1(torch.cat([enc1, dec1], dim=1))
        
        # Final output
        return self.sigmoid(self.final(up1))


# Example Usage
if __name__ == "__main__":
    # Initialize U-Net
    model = binarizer(in_channels=1, out_channels=1)  # For grayscale input and binary output
    print(model)
    
    # Test with a random tensor
    x = torch.randn(1, 1, 512, 512)  # Batch size of 1, single channel, 256x256 image
    y = model(x)
    print(y.shape)  # Output shape should match input shape except for channel dimension
