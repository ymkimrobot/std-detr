import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Backbone(nn.Module):

    def __init__(self, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Backbone, self).__init__()

        block = BasicBlock
        layers = [2,2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128*4, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        #cam
        self.dropout_cam = torch.nn.Dropout2d(0.5)
        self.fc_cam = nn.Conv2d(128, 2, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(128 + 3, 128, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        # padding if #channel is not 4.
        x_origin = x
        size = x.size()
        N, C, H, W = x.view(-1, x.shape[2], x.shape[3], x.shape[4]).size()

        if x.size(-3)==2:
            padded = x.new(x.size()).zero_()
            x = torch.cat([x, padded], 1)

        x = x.view(-1, *x.size()[-3:])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x = self.layer2(x2)

        # feature 중첩
        f_1 = x[:,0:128, :,:]
        f_2 = x[:,128:256, :,:]
        f_3 = x[:, 256:384, :, :]
        f_4 = x[:, 384:512, :, :]
        x = f_1 + f_2 + f_3 + f_4
        # (f_1 > f_2)
        #method1
        # f_2 = torch.max(f_1, f_2)
        # f_3 = torch.max(f_2, f_3)
        # x = torch.max(f_3, f_4)

        #method2
        # f_1 = torch.roll(f_1, shifts=5, dims=2)
        # f_1 = torch.roll(f_1, shifts=-5, dims=3)
        # f_2 = torch.roll(f_2, shifts=-5, dims=2)
        # f_2 = torch.roll(f_2, shifts=-5, dims=3)
        # f_3 = torch.roll(f_3, shifts=5, dims=2)
        # f_3 = torch.roll(f_3, shifts=5, dims=3)
        # f_4 = torch.roll(f_4, shifts=-5, dims=2)
        # f_4 = torch.roll(f_4, shifts=5, dims=3)
        #
        # f_2 = torch.max(f_1, f_2)
        # f_3 = torch.max(f_2, f_3)
        # x = torch.max(f_3, f_4)

        #cam

        cam = self.fc_cam(self.dropout_cam(x))
        # n,c,h,w = cam.size()


        # with torch.no_grad():
        #     cam_d = F.relu(cam.detach())
        #     cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
        #     cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
        #     cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
        #     cam_max = torch.max(cam_d_norm, dim=1, keepdim=True)[0]
        #     cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0



        # x_s = F.interpolate(x_origin.squeeze(0), (64, 64), mode='bicubic', align_corners=True)
        # f = torch.cat([x_s, x1, x2], dim=1)
        # f = torch.cat([x1, x2], dim=1)
        # n,c,h,w = f.size()

        # f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        # f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        # cam_rv = F.interpolate(f, (32, 32), mode='bilinear', align_corners=True)
        # cam_rv = F.interpolate(self.PCM(cam_d, f), (32,32), mode='bicubic', align_corners=True)
        # cam = F.interpolate(cam, (32,32), mode='bilinear', align_corners=True)
        #temp
        # cam_rv = cam
        x = x.view(*size[:-3], *x.size()[-3:])




        return x, cam

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv