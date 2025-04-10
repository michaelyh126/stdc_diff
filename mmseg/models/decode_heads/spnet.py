import spconv.pytorch as spconv
from torch import nn
import torch
from .refine_decode_head import RefineBaseDecodeHead

class SparseBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, indice_key=None):
        super(SparseBottleneck, self).__init__()
        self.spconv1=spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=1,indice_key=indice_key),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),)
        self.spconv2=spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=mid_channels,out_channels=mid_channels,kernel_size=3,indice_key=indice_key),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),)
        self.spconv3=spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,indice_key=indice_key),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),)


        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = spconv.SparseSequential(
                spconv.SubMConv2d(in_channels, out_channels, kernel_size=1, stride=stride, indice_key=indice_key),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x=self.spconv1(x)
        x=self.spconv2(x)
        out=self.spconv3(x)
        # Element-wise addition for residual connection
        out =out.replace_feature(residual.features+out.features)
        return out




class SparseResNet50(RefineBaseDecodeHead):
    def __init__(self, shape,**kwargs):
        super(SparseResNet50,self).__init__(**kwargs)
        self.shape = shape
        num_classes=self.num_classes

        # Initial Convolution Layer
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(3, 32, kernel_size=7, stride=2, padding=3, indice_key="subm0"),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Define each block group in ResNet
        self.layer1 = self._make_layer(32, 64, 128, num_blocks=3, stride=1, indice_key="subm1")
        self.layer2 = self._make_layer(128, 256, 128, num_blocks=3, stride=1, indice_key="subm2")
        # self.layer3 = self._make_layer(512, 256, 1024, num_blocks=4, stride=2, indice_key="subm3")
        # self.layer4 = self._make_layer(1024, 512, 2048, num_blocks=3, stride=2, indice_key="subm4")

        # Fully connected layer for classification
        self.classifier = spconv.SparseSequential(
            spconv.SubMConv2d(128, num_classes, kernel_size=1, indice_key="subm5")
        )

    def _make_layer(self, in_channels, mid_channels, out_channels, num_blocks, stride, indice_key):
        layers = []
        layers.append(SparseBottleneck(in_channels, mid_channels, out_channels, stride, indice_key))
        for _ in range(1, num_blocks):
            layers.append(SparseBottleneck(out_channels, mid_channels, out_channels, stride=1, indice_key=indice_key))
        # return spconv.SparseSequential(*layers)
        return nn.Sequential(*layers)

    def forward(self, features, coors, batch_size):
        coors = coors.int()  # spconv requires integer coordinates
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.classifier(x)

        return x

class SpNet(RefineBaseDecodeHead):
    def __init__(self, shape,**kwargs):
        super(SpNet,self).__init__(**kwargs)
        # self.net = spconv.SparseSequential(
        #     spconv.SparseConv2d(in_channels=32,out_channels=64,stride=1,kernel_size=3),
        #     # nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #
        #     spconv.SubMConv2d(64, 64, 3, indice_key="subm0"),
        #     # nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     # when use submanifold convolutions, their indices can be shared to save indices generation time.
        #     spconv.SubMConv2d(64, 64, 3, indice_key="subm0"),
        #     # nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     spconv.SparseConvTranspose2d(64, 64, 3, 2),
        #     # nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
        #     nn.Conv2d(64, 64, 3),
        #     # nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        self.shape = shape

        self.spconv1=spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=3,out_channels=32,kernel_size=3,indice_key="subm0"),
            nn.BatchNorm1d(32),
            nn.ReLU(),)
        self.spconv2=spconv.SparseSequential(
            spconv.SubMConv2d(32, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
          )
        self.spconv3=spconv.SparseSequential(
            spconv.SubMConv2d(64, 128, 3, indice_key="subm0"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
          )
        self.spconv4=spconv.SparseSequential(
            spconv.SubMConv2d(128, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
          )
        self.spconv5=spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, 3, indice_key="subm0"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
          )
        self.spconv6=spconv.SparseSequential(
            spconv.SubMConv2d(32, self.num_classes, 3, indice_key="subm0"),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
          )




    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        x1=self.spconv1(x)
        x2=self.spconv2(x1)
        x3=self.spconv3(x2)
        x4=self.spconv4(x3)
        x5=self.spconv5(x4)
        x6=self.spconv6(x5)


        return x6


def get_coordsandfeatures(input_tensor):
    first_channel = input_tensor[:, 0, :, :]
    coords = torch.nonzero(first_channel, as_tuple=False)
    coords=coords.cuda()
    b = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]

    # 利用索引操作提取特征
    features_tensor = input_tensor[b, :, y, x].cuda()
    return coords,features_tensor

def main():
    # input_tensor = torch.randn(1, 3, 128, 128)  # (B, C, H, W)
    # sparsity = 0.1
    # mask = torch.rand_like(input_tensor) > sparsity  # 生成一个布尔掩码
    # input_tensor[mask] = 0  # 使用掩码将大部分元素置为零
    input_tensor = torch.rand(8, 1, 3, 3)
    b,c,h,w=input_tensor.shape
    mask = torch.rand_like(input_tensor) > 0.3
    input_tensor[mask] = 0
    input_tensor = input_tensor.repeat(1, 3, 1, 1)
    coords,features=get_coordsandfeatures(input_tensor)

    input_shape = (h,w)

    features = features.cuda()
    coords = coords.cuda()
    # features = torch.rand(num_points, 3).cuda()  # 32个特征
    # coors = torch.randint(0, 128, (num_points, 3)).cuda()

    # coors[:, 0] = batch_size - 1  # Set all points to the same batch index for this example

    # Initialize the model and move it to GPU
    model = SpNet(input_shape,2).cuda()

    # Forward pass
    output = model(features, coords, b)
    output=output.dense()
    num_positive_elements = (output != 0).sum().item()
    # Print the output shape
    print("Output shape:", output.shape)  # Convert to dense format to see the shape

if __name__ == "__main__":
    main()
