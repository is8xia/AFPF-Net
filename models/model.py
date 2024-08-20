import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18


def ConvBNReLU(in_channel, out_channel, kernel_size=3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )


class Redection(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Redection, self).__init__()
        self.reduce = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel_size=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3),
            ConvBNReLU(out_ch, out_ch, kernel_size=3)
        )

    def forward(self, x):
        y = self.reduce(x)
        return y


class FDE(nn.Module):
    def __init__(self, channel):
        super(FDE, self).__init__()
        self.conv_cat = ConvBNReLU(channel * 2, channel)
        self.sam = SpatialAttention()
        self.cam = ChannelAttention(channel * 2)
        self.conv_enh1 = ConvBNReLU(channel, channel)
        self.conv_enh2 = ConvBNReLU(channel, channel, stride=2)
        self.conv_enh3 = ConvBNReLU(channel, channel)
        self.conv_diff_enh1 = ConvBNReLU(channel, channel)
        self.conv_diff_enh2 = ConvBNReLU(channel, channel)

    def forward(self, x, y, mask=None):
        feat_sub = self.conv_enh1(torch.abs(x - y))
        feat_weight = self.sam(feat_sub)
        if mask is not None:
            mask = self.sam(self.conv_enh2(mask))
            feat_weight = (feat_weight + mask) / 2

        x_result = self.conv_diff_enh1(x.mul(feat_weight) + x)
        y_result = self.conv_diff_enh1(y.mul(feat_weight) + y)

        x_f = torch.cat([x_result, y_result], dim=1)
        x_f = self.conv_cat(self.cam(x_f) * x_f) + feat_sub
        result = self.conv_enh3(x_f)

        return result


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class FeatureEnhancementUnit(nn.Module):
    def __init__(self, channel):
        super(FeatureEnhancementUnit, self).__init__()
        self.channel = channel

        self.conv_cat1 = ConvBNReLU(self.channel * 2, self.channel)
        self.ca1 = ChannelAttention(self.channel * 2)

        self.conv_cat2 = ConvBNReLU(self.channel * 2, self.channel)
        self.ca2 = ChannelAttention(self.channel * 2)

        self.conv_spilt1 = ConvBNReLU(self.channel * 2, self.channel * 2, kernel_size=1)
        self.conv_spilt2 = ConvBNReLU(self.channel * 2, self.channel * 2, kernel_size=1)

        self.conv_fusion = ConvBNReLU(self.channel * 3, self.channel)

        self.mask_generation_x = nn.Sequential(
            ConvBNReLU(self.channel, self.channel),
            nn.Conv2d(self.channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.mask_generation_y = nn.Sequential(
            ConvBNReLU(self.channel, self.channel),
            nn.Conv2d(self.channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        f_cat = torch.cat([x, y], dim=1)
        f_cat_1 = self.conv_spilt1(f_cat)
        f_cat_ca = self.ca1(f_cat_1) * f_cat_1
        fca_1 = self.conv_cat1(f_cat_ca)

        y_mask = self.mask_generation_y(y)
        x_mask = self.mask_generation_x(x)
        reverse_att = 1 - x_mask
        fca_2 = reverse_att * y

        uncertainty_att = (1 - x_mask) * y_mask + (1 - y_mask) * x_mask
        f_cat_2 = self.conv_spilt2(f_cat)
        f_cat_2enh = f_cat_2 * uncertainty_att + f_cat_2
        fca_3 = self.ca1(f_cat_2enh) * f_cat_2enh
        fca_3 = self.conv_cat2(fca_3)

        f = torch.cat([fca_1, fca_2, fca_3], dim=1)
        f = self.conv_fusion(f)

        return f


class FusionBlock(nn.Module):
    def __init__(self, channel):
        super(FusionBlock, self).__init__()
        self.channel = channel

        self.feu = FeatureEnhancementUnit(self.channel)

    def forward(self, d5, d4):

        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        out = self.feu(d5, d4)

        return out


class BaseNet(nn.Module):
    def __init__(self, channel=64):
        super(BaseNet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.mid_d = channel
        self.rec_ch = [64, 128, 256, 512]

        self.reduction_1 = Redection(self.rec_ch[0], self.mid_d)
        self.reduction_2 = Redection(self.rec_ch[1], self.mid_d)
        self.reduction_3 = Redection(self.rec_ch[2], self.mid_d)
        self.reduction_4 = Redection(self.rec_ch[3], self.mid_d)

        self.reduction_11 = Redection(self.rec_ch[0], self.mid_d)
        self.reduction_22 = Redection(self.rec_ch[1], self.mid_d)
        self.reduction_33 = Redection(self.rec_ch[2], self.mid_d)
        self.reduction_44 = Redection(self.rec_ch[3], self.mid_d)

        self.fde1 = FDE(self.mid_d)
        self.fde2 = FDE(self.mid_d)
        self.fde3 = FDE(self.mid_d)
        self.fde4 = FDE(self.mid_d)

        self.fsb1 = FusionBlock(self.mid_d)
        self.fsb2 = FusionBlock(self.mid_d)
        self.fsb3 = FusionBlock(self.mid_d)

        self.fsb11 = FusionBlock(self.mid_d)
        self.fsb22 = FusionBlock(self.mid_d)

        self.fsb111 = FusionBlock(self.mid_d)
        self.conv_out = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, t1_fea1, t1_fea2, t1_fea3, t1_fea4 = self.backbone.base_forward(x1)
        x2_1, t2_fea1, t2_fea2, t2_fea3, t2_fea4 = self.backbone.base_forward(x2)
        # feature difference

        t1_fea1 = self.reduction_1(t1_fea1)
        t1_fea2 = self.reduction_2(t1_fea2)
        t1_fea3 = self.reduction_3(t1_fea3)
        t1_fea4 = self.reduction_4(t1_fea4)

        t2_fea1 = self.reduction_11(t2_fea1)
        t2_fea2 = self.reduction_22(t2_fea2)
        t2_fea3 = self.reduction_33(t2_fea3)
        t2_fea4 = self.reduction_44(t2_fea4)

        d2 = self.fde1(t1_fea1, t2_fea1)
        d3 = self.fde2(t1_fea2, t2_fea2, d2)
        d4 = self.fde3(t1_fea3, t2_fea3, d3)
        d5 = self.fde4(t1_fea4, t2_fea4, d4)

        # step 1
        d2 = self.fsb1(d3, d2)
        d3 = self.fsb2(d4, d3)
        d4 = self.fsb3(d5, d4)

        # step 2
        d2 = self.fsb11(d3, d2)
        d3 = self.fsb22(d4, d3)

        # step 3
        d2 = self.fsb111(d3, d2)
        d2 = self.conv_out(d2)

        # decoder
        mask = F.interpolate(d2, x1.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.sigmoid(mask)

        return mask

if __name__ == '__main__':
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)
    model = BaseNet(64)
    x1 = model(img1, img2)
    print(x1.size())
