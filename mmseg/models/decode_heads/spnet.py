import spconv.pytorch as spconv
from torch import nn
import torch
from .refine_decode_head import RefineBaseDecodeHead


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
