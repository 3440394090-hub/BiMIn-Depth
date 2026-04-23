# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
# from pyexpat import features
from torchvision.models.resnet import Bottleneck
import numpy as np
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class RefinedResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(RefinedResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        self.SCAM_1 = SCAM(n_feats=256)
        self.SCAM_2 = SCAM(n_feats=512)
        pretrained_resnet = resnet50(weights="IMAGENET1K_V1")
        self.custom_layer3 = Layer3WithODRA()
        load_pretrained_layer3(self.custom_layer3, pretrained_resnet)

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        x1 = self.encoder.layer1(self.encoder.maxpool(self.features[-1]))
        x1 = x1 + self.SCAM_1(x1)
        self.features.append(x1)

        x2 = self.encoder.layer2(self.features[-1])
        x2 = x2 + self.SCAM_2(x2)
        self.features.append(x2)
        self.features.append(self.custom_layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=512):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        # for res50
        self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 512, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 256, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16)

        # self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        # x_d5 = self.up5(x_d4, features[0])
        # out = self.conv3(x_d5)
        out = self.conv3(x_d4)
        return out


class RefinedEncoderDecoder(nn.Module):
    def __init__(self, num_layers=50, num_features=512, model_dim=32):
        super(RefinedEncoderDecoder, self).__init__()
        self.encoder = RefinedResnetEncoder(num_layers = num_layers, pretrained = True, num_input_images=1)
        self.decoder = DecoderBN(num_features=num_features, num_classes=model_dim, bottleneck_features=2048)
    def forward(self, x, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, **kwargs)

class Refined50EncoderDecoder(nn.Module):
    def __init__(self, model_dim=128):
        super(Refined50EncoderDecoder, self).__init__()
        # for Res50
        self.encoder = RefinedResnetEncoder(num_layers = 50, pretrained = True, num_input_images=1)
        self.decoder = DecoderBN(num_features=512, num_classes=model_dim, bottleneck_features=2048)

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, **kwargs)


class SCAM(nn.Module):
    def __init__(self, n_feats):
        super(SCAM, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        # 平滑激活函数（替换ReLU）
        self.act = nn.GELU()
        self.hardsigmoid = nn.Hardsigmoid()
        # 防梯度放大
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        # v_range = self.relu(self.conv_max(v_max))
        v_range = self.act(self.conv_max(v_max))
        # c3 = self.relu(self.conv3(v_range))
        c3 = self.act(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        # m = self.sigmoid(c4)
        m = self.hardsigmoid(c4)

        return self.scale * (x * m)

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat([
            torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1).unsqueeze(1)
        ], dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        return self.conv(x).sigmoid()

class ODRA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.zpool = ZPool()
        self.gateH = AttentionGate()
        self.gateW = AttentionGate()
        self.gateS = AttentionGate()
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_norm = self.bn(x)  # 先归一化
        x_perm1 = x_norm.permute(0, 2, 1, 3)
        x_out1 = self.gateH(self.zpool(x_perm1)) * x_perm1
        x_out1 = x_out1.permute(0, 2, 1, 3)

        x_perm2 = x_norm.permute(0, 3, 2, 1)
        x_out2 = self.gateW(self.zpool(x_perm2)) * x_perm2
        x_out2 = x_out2.permute(0, 3, 2, 1)

        x_out3 = self.gateS(self.zpool(x_norm)) * x_norm

        out = (x_out1 + x_out2 + x_out3) / 3
        return x + self.gamma * out



# ---------------- 自定义 layer3 ----------------
class Layer3WithODRA(nn.Module):
    def __init__(self):
        super().__init__()
        # 第1个 Bottleneck stride=2 降采样
        self.block1 = Bottleneck(512, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        ))
        self.ta1 = ODRA(1024)

        self.block2 = Bottleneck(1024, 256)
        # self.ta2 = ODRA()

        self.block3 = Bottleneck(1024, 256)
        self.ta3 = ODRA(1024)

        self.block4 = Bottleneck(1024, 256)
        # self.ta4 = ODRA()

        self.block5 = Bottleneck(1024, 256)
        # self.ta5 = ODRA()

        self.block6 = Bottleneck(1024, 256)
        self.ta6 = ODRA(1024)

    def forward(self, x):
        x = self.ta1(self.block1(x))
        x = self.block2(x)
        x = self.ta3(self.block3(x))
        x = self.block4(x)
        x = self.block5(x)
        x = self.ta6(self.block6(x))
        return x

# ---------------- 初始化 Layer3WithTriplet 权重 ----------------
def load_pretrained_layer3(custom_layer3, pretrained_resnet):
    # 获取官方 layer3 权重
    pretrained_layer3 = pretrained_resnet.layer3
    pretrained_dict = pretrained_layer3.state_dict()
    custom_dict = custom_layer3.state_dict()

    # 只匹配相同名字的参数（Bottleneck 部分）
    matched_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict}
    custom_dict.update(matched_dict)
    custom_layer3.load_state_dict(custom_dict)
    print(" Layer3 Bottleneck 权重已成功加载预训练模型（ODRA 随机初始化）")
