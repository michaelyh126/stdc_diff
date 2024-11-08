from ..builder import HEADS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from .refine_decode_head import RefineBaseDecodeHead

@HEADS.register_module()
class DiffHead(RefineBaseDecodeHead):
    def __init__(self,**kwargs):
        super(DiffHead, self).__init__(**kwargs)
        # self.starnet=starnet_s3()
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(self.channels, 3, kernel_size=3, stride=1, padding=1),
        # )


        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.in_channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.in_channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.in_channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        #
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1),

        )




    def forward(self,input):
        x1=self.conv_layer1(input)
        x2=self.conv_layer2(x1)
        x3=self.conv_layer3(x2)
        x4=self.conv_layer4(x3)

        # outputs=self.starnet(input)
        # output=self.conv_layer(outputs[3])

        return x3,x4


    #     if type == "cat":
    #         block = CatBottleneck
    #     elif type == "add":
    #         block = AddBottleneck
    #     self.in_channels = in_channels
    #     self.features = self._make_layers(base, layers, block_num, block)
    #     self.x2 = nn.Sequential(self.features[:1])
    #     self.x4 = nn.Sequential(self.features[1:2])
    #     self.x8 = nn.Sequential(self.features[2:4])
    #     self.x16 = nn.Sequential(self.features[4:6])
    #     if pretrain_model:
    #         print('use pretrain model {}'.format(pretrain_model))
    #         self.init_weight(pretrain_model)
    #     else:
    #         self.init_params()
    #
    # def init_weight(self, pretrain_model):
    #
    #     state_dict = torch.load(pretrain_model)["state_dict"]
    #     self_state_dict = self.state_dict()
    #     for k, v in state_dict.items():
    #         if k == 'features.0.conv.weight' and self.in_channels != 3:
    #             v = torch.cat([v, v], dim=1)
    #         self_state_dict.update({k: v})
    #     self.load_state_dict(self_state_dict, strict=False)
    #
    # def init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #
    # def _make_layers(self, base, layers, block_num, block):
    #     features = []
    #     features += [ConvX(self.in_channels, base//2, 3, 2)]
    #     features += [ConvX(base//2, base, 3, 2)]
    #
    #     for i, layer in enumerate(layers):
    #         for j in range(layer):
    #             if i == 0 and j == 0:
    #                 features.append(block(base, base*4, block_num, 2))
    #             elif j == 0:
    #                 features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
    #             else:
    #                 features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
    #
    #     return nn.Sequential(*features)
    #
    # def forward(self, x, cas3=False):
    #     feat2 = self.x2(x)
    #     feat4 = self.x4(feat2)
    #     feat8 = self.x8(feat4)
    #     feat16 = self.x16(feat8)
    #     if cas3:
    #         return feat4, feat8, feat16
    #     else:
    #         return feat8, feat16
