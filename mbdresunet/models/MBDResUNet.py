import torch.nn as nn
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm3d
from thop import profile
from thop import clever_format
from torch.nn.parameter import Parameter
# try:
#     from .sync_batchnorm import SynchronizedBatchNorm3d
# except:
#     pass

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in,norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h

class SCA3D(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel//2, int(channel // (2*reduction))),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // (2*reduction)), channel//2))
        # self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,stride=1, padding=0, bias=False)
        self.spatial_se = nn.Conv3d(channel//2, 1,  kernel_size=(1,1,1), stride=1, padding=0, bias=False)

    def channel_shuffle(self,x, groups):

        b, c, h, w, d = x.shape

        x = x.reshape(b, groups, -1, h, w, d)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, h, w, d)

        return x

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        x_0, x_1 = x.chunk(2, dim=1)
        chn_se = self.avg_pool(x_0).view(bahs, chs//2)
        # print(chn_se.shape)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs//2, 1, 1, 1))
        chn_se = torch.mul(x_0, chn_se)
        spa_se = torch.sigmoid(self.spatial_se(x_1))
        spa_se = torch.mul(x_1, spa_se)
        out = torch.cat([chn_se, spa_se], dim=1)
        # net_out = spa_se + x + chn_se
        out = out + x
        # print(out.shape)
        # out = out.reshape(bahs, -1, _, _, _)
        out = self.channel_shuffle(out,2)
        return out

class MBResBlock(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):

        super(MBResBlock, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = DilatedConv3DBlock(num_in // 4, num_mid, kernel_size=(3, 3, 3), stride=stride, g=g,
                                               d=(d[0], d[0], d[0]), norm=norm)  # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_mid, num_out, kernel_size=(3, 3, 1), stride=1, g=g,
                                               d=(d[1], d[1], 1), norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_out, num_out, kernel_size=1, stride=1, norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):

        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv3x3x3_m1(x1)
        x3 = self.conv3x3x3_m2(x2)
        x4 = self.conv1x1x1_in2(x3)

        shortcut = x

        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut

class MBDResBlock(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(MBDResBlock, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        # self.weight1 = 1
        # self.weight2 = 1
        # self.weight3 = 1

        num_mid = num_in if num_in <= num_out else num_out


        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1, 2, 3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_in // 4, num_mid, kernel_size=(3, 3, 3), stride=stride, g=g,
                                   d=(dilation[i], dilation[i], dilation[i]), norm=norm)

            )

        # It has not Dilated operation
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_mid, num_out, kernel_size=(3, 3, 1), stride=(1, 1, 1), g=g,
                                               d=(1, 1, 1), norm=norm)

        self.conv1x1x1_in2 = Conv3d_Block(num_out, num_out, kernel_size=1, stride=1, norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)


    def forward(self, x):

        x1 = self.conv1x1x1_in1(x)
        x2 = self.weight1 * self.conv3x3x3_m1[0](x1) + self.weight2 * self.conv3x3x3_m1[1](x1) + self.weight3 * \
             self.conv3x3x3_m1[2](x1)
        x3 = self.conv3x3x3_m2(x2)
        x4 = self.conv1x1x1_in2(x3)
        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        return x4 + shortcut


class MBResUNet(nn.Module): #

    def __init__(self, c=4,n=32,channels=128,groups = 16,norm='bn', num_classes=4):
        super(MBResUNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=2, bias=False)# H//2
        self.encoder_block2 = nn.Sequential(
            MBResBlock(n, channels, g=groups, stride=2, norm=norm),# H//4 down
            MBResBlock(channels, channels, g=groups, stride=1, norm=norm),
            MBResBlock(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MBResBlock(channels, channels*2, g=groups, stride=2, norm=norm), # H//8
            MBResBlock(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MBResBlock(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(# H//8,channels*4
            MBResBlock(channels*2, channels*3, g=groups, stride=2, norm=norm), # H//16
            MBResBlock(channels*3, channels*3, g=groups, stride=1, norm=norm),
            MBResBlock(channels*3, channels*2, g=groups, stride=1, norm=norm),
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.decoder_block1 = MBResBlock(channels*2+channels*2, channels*2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.decoder_block2 = MBResBlock(channels*2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.decoder_block3 = MBResBlock(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.sca1 = SCA3D(32)
        self.sca = SCA3D(4)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x = self.sca(x)
        x1 = self.encoder_block1(x)# H//2 down
        x2 = self.encoder_block2(x1)# H//4 down
        x3 = self.encoder_block3(x2)# H//8 down
        x4 = self.encoder_block4(x3) # H//16
        # Decoder
        y1 = self.upsample1(x4)# H//8
        y1 = torch.cat([x3,y1],dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)# H//4
        y2 = torch.cat([x2,y2],dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)# H//2
        y3 = torch.cat([x1,y3],dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self,'softmax'):
            y4 = self.softmax(y4)
        return y4


class MBDResUNet(MBResUNet): # softmax
    def __init__(self, c=4,n=32,channels=128, groups=16,norm='bn', num_classes=4):
        super(MBDResUNet, self).__init__(c,n,channels,groups, norm, num_classes)

        self.encoder_block2 = nn.Sequential(
            MBDResBlock(n, channels, g=groups, stride=2, norm=norm,dilation=[1,2,3]),# H//4 down
            MBDResBlock(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3]), # Dilated Conv 3
            MBDResBlock(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )

        self.encoder_block3 = nn.Sequential(
            MBDResBlock(channels, channels*2, g=groups, stride=2, norm=norm,dilation=[1,2,3]), # H//8
            MBDResBlock(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3]),# Dilated Conv 3
            MBDResBlock(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1,4,128,128,128),device=device) # [bsize,channels,Height,Width,Depth]
    model = MBDResUNet(c=4, groups=8, norm='sync_bn', num_classes=4)
    model.cuda(device)
    y = model(x)
    print(y.shape)
    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print("Model FLOPs: ", macs)
    print("Model Params:", params)
