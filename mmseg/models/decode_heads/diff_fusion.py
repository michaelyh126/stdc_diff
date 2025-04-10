import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self,in_channels=2):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size, stride=1,padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x_compress = self.compress(x)
        # save_heatmap(x_compress[1].detach().cpu().numpy(), filename='att_heatmap_compress0',save_dir='D:\deep_learning\ISDNetV2\diff_dir\heatmap', channel=1)
        # save_heatmap(x_compress[1].detach().cpu().numpy(), filename='att_heatmap_compress1',save_dir='D:\deep_learning\ISDNetV2\diff_dir\heatmap', channel=2)

        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        # save_heatmap(scale[0].detach().cpu().numpy(), filename='att_heatmap_scale',save_dir='D:\deep_learning\ISDNetV2\diff_dir\heatmap', channel=0)
        return  scale



class DiffFusion(nn.Module):
    def __init__(self,in_channels,channels,sp_channel,cp_channel ):
        super(FeatureFusionModule, self).__init__()
        in_chan=in_channels
        out_chan=channels
        self.convblk = ConvBNReLU(cp_channel*2, cp_channel, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.convblk2 = ConvBN(2, 1, ks=1, stride=1, padding=0)
        self.convblk3 = ConvBN(1, cp_channel, ks=1, stride=1, padding=0)
        self.conv_downsample = ConvBN(cp_channel,sp_channel,ks=1,stride=1,padding=0)
        self.cls_seg=ConvBN(128,512,ks=1,stride=1,padding=0)
        # self.conv_mul_fsp_score = ConvBN(1,1,ks=1,stride=1,padding=0)
        # TODO
        self.sg=SpatialGate()
        self.channelpool=ChannelPool()

    def forward(self,fsp,fcp,mask=None,test_fcp_fuse=False):
        fcp=self.conv_downsample(fcp)
        fcp = resize(input=fsp,size=fsp.shape[2:],mode='bilinear',align_corners=None)
        fcp_fuse=fcp_mask+fsp_mask
        if test_fcp_fuse==True:
            return fcp_fuse

        fcat = torch.cat([fsp, fcp_fuse], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_channel = feat_atten + feat
        sg_score=self.sg(feat_channel)
        feat_out=feat_channel*sg_score
        # feat_out=self.cls_seg(feat_out)
        return fsp,fcp_fuse,feat_out


class FeatureFusionModule(nn.Module):
    def __init__(self,in_channels,channels,sp_channel,cp_channel ):
        super(FeatureFusionModule, self).__init__()
        in_chan=in_channels
        out_chan=channels
        self.convblk = ConvBNReLU(cp_channel*2, cp_channel, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.convblk2 = ConvBN(2, 1, ks=1, stride=1, padding=0)
        self.convblk3 = ConvBN(1, cp_channel, ks=1, stride=1, padding=0)
        self.conv_downsample = ConvBN(sp_channel,cp_channel,ks=1,stride=1,padding=0)
        self.cls_seg=ConvBN(128,512,ks=1,stride=1,padding=0)
        # self.conv_mul_fsp_score = ConvBN(1,1,ks=1,stride=1,padding=0)
        # TODO
        self.sg=SpatialGate()
        self.channelpool=ChannelPool()

    def forward_without_att(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        return feat

    def forward(self,fsp,fcp,mask=None,test_fcp_fuse=False):
        mask=resize(input=mask, size=fsp.shape[2:], mode='bilinear', align_corners=None)
        mask=torch.sigmoid(mask)
        mask = (mask > 0.5).float()
        inverse_mask=(mask <= 0.5).float()
        fsp=self.conv_downsample(fsp)
        fsp_mask=torch.mul(fsp,mask)
        # fsp_score=self.channelpool(fsp_mask)
        # fsp_score=self.convblk2(fsp_score)
        # # mul_fsp_score=self.conv_mul_fsp_score(fsp)
        # # fsp_score_mean=torch.mean(fsp_score, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        # # fsp_score = torch.where(fsp_score == 0, fsp_score_mean, fsp_score)
        # fsp_score = self.convblk3(fsp_score)
        # fsp_score=torch.sigmoid(fsp_score)
        fcp = resize(input=fcp,size=fsp.shape[2:],mode='bilinear',align_corners=None)
        fcp_mask=torch.mul(fcp,inverse_mask)
        fcp_fuse=fcp_mask+fsp_mask
        # fcp_fuse=fcp*fsp_score
        # fsp_fuse=fsp*fsp_score
        if test_fcp_fuse==True:
            return fcp_fuse

        fcat = torch.cat([fsp, fcp_fuse], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_channel = feat_atten + feat
        sg_score=self.sg(feat_channel)
        feat_out=feat_channel*sg_score
        # feat_out=self.cls_seg(feat_out)
        return fsp,fcp_fuse,feat_out

class FFM(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FFM, self).__init__()
        self.convblk = ConvBNReLU(in_chan*2, out_chan, ks=1, stride=1, padding=0)
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)


    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out
