import spconv.pytorch as spconv
from torch import nn
import torch
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.net = spconv.SparseSequential(
        #     spconv.SparseConv2d(in_channels=32,out_channels=64,stride=1,kernel_size=3,padding=1),
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
            spconv.SparseConv2d(in_channels=32,out_channels=64,stride=1,kernel_size=3,padding=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),)
        self.spconv2=spconv.SparseSequential(
            spconv.SubMConv2d(64, 64, 3, indice_key="subm0"),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
          )
        self.spconv3=spconv.SparseSequential(
            spconv.SubMConv2d(64, 64, 3, indice_key="subm0"),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
          )
        self.spconv4=spconv.SparseSequential(
            spconv.SubMConv2d(64, 64, 3, indice_key="subm0"),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
          )
        self.spconv4=spconv.SparseSequential(
            spconv.SparseConvTranspose2d(64, 64, 3, 2),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
          )
        self.todense=spconv.ToDense()
        self.spconv5=spconv.SparseSequential(
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
            nn.Conv2d(64, 64, 3),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
          )



    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        x1=self.spconv1(x)
        x2=self.spconv2(x1)
        x3=self.spconv3(x2)
        x4=self.spconv4(x3)
        x_dense=self.todense(x4)
        x5=self.spconv5(x_dense)
        return x5# .dense()


def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-capable GPU and the correct drivers installed.")

    # Define input shape (depth, height, width)
    input_shape = (128, 128)  # Shape for the sparse tensor
    batch_size = 1

    # Create random features and coordinates
    num_points = 1000  # Number of sparse points
    features = torch.rand(num_points, 32).cuda()  # Move features to GPU
    coors = torch.randint(0, 128, (num_points, 3)).cuda()  # Move coordinates to GPU

    # The first column of coordinates is the batch index, and the next three are spatial coordinates
    coors[:, 0] = batch_size - 1  # Set all points to the same batch index for this example

    # Initialize the model and move it to GPU
    model = ExampleNet(input_shape).cuda()

    # Forward pass
    output = model(features, coors, batch_size)

    # Print the output shape
    print("Output shape:", output.dense().shape)  # Convert to dense format to see the shape

if __name__ == "__main__":
    main()
