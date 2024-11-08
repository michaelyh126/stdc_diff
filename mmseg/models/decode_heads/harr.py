import torch
import torch.nn as nn
import torch.nn.functional as F

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def separate(x):
    in_batch, in_channel, in_height, in_width = x.size()
    out_channel=in_channel//4
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    return x1,x2,x3,x4


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class CNN(nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)  # BatchNorm after first conv layer

    def forward(self, x):
        x = self.bn1(self.conv1(x))  # Apply BN after the first conv
        return x


class HarrDown(nn.Module):
    def __init__(self):
        super(HarrDown, self).__init__()
        self.cnn1=CNN(channels=12).cuda()
        self.cnn2=CNN(channels=48).cuda()
        self.dwt_layer = DWT().cuda()

    def forward(self, x):
        x1=self.dwt_layer(x)
        x2=self.cnn1(x1)
        x21,x22,x23,x24=separate(x2)
        dwt1_list=[x21,x22,x23,x24]
        dwt2_list=[]
        for xi in dwt1_list:
            dwt_result = self.dwt_layer(xi)
            dwt2_list.append(dwt_result)
        x3=torch.cat((dwt2_list[0],dwt2_list[1],dwt2_list[2],dwt2_list[3]),1)
        x4=self.cnn2(x3)
        return x4

class HarrUp(nn.Module):
    def __init__(self):
        super(HarrUp, self).__init__()
        self.cnn3=CNN(channels=128).cuda()
        self.iwt_layer = IWT().cuda()
        self.conv = nn.Conv2d(in_channels=8, out_channels=128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(128)  # BatchNorm after first conv layer

    def forward(self, x):
        x1=self.cnn3(x)
        x11,x12,x13,x14=separate(x1)
        iwt1_list=[x11,x12,x13,x14]
        iwt2_list=[]
        iwt3_list=[]
        for xi in iwt1_list:
            iwt1_result=self.iwt_layer(xi)
            iwt2_list.append(iwt1_result)
        for xj in iwt2_list:
            iwt2_result=self.iwt_layer(xj)
            iwt3_list.append(iwt2_result)
        x2=torch.cat((iwt3_list[0],iwt3_list[1],iwt3_list[2],iwt3_list[3]),1)
        x3=self.bn(self.conv(x2))
        return x3









if __name__ == '__main__':
    input_tensor = torch.rand(1, 64, 32, 32).cuda()
    down=DWT().cuda()
    up=IWT().cuda()
    # down=HarrDown().cuda()
    out=up(input_tensor)
    print('end')

