
import torch
import torch.nn as nn
import torch.nn.functional as F


from nets import MASA

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone="MASA", pretrained=True, downsample_factor=16):
        super(Decoder, self).__init__()
        if backbone == "MASA":
            self.backbone = MASA.MAXIM_backbone()
            in_channels =512
            low_level_channels = 128
        # elif backbone == "xception":
        #     self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
        #     # print(xception())
        #     in_channels = 2048
        #     low_level_channels = 256

        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.GELU()
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        low_level_features,x= self.backbone(x)

        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)

        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
