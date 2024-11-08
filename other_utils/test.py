import torch
import torch.nn.functional as F
from mmseg.datasets.pipelines.loading import LoadImageFromFile
import mmcv


if __name__ == '__main__':
    loading=LoadImageFromFile()
    loading.file_client = mmcv.FileClient(**loading.file_client_args)
    f=loading.file_client.get('D:\deep_learning\ISDNet-main\dataset_island512\imgs\\train\hjd_0_46.png')
    f=loading.file_client.get('D:\deep_learning\ISDNet-main\dataset_island512\imgs\\train\hjd_24_45.tif')
    f=loading.file_client.get('D:\deep_learning\ISDNet-main\\root_path\imgs\\vienna3.tif')
    logits = torch.rand((2, 2, 2000, 2000), dtype=torch.float32)
    random_tensor = torch.randint(low=0, high=2, size=(2, 2000, 2000), dtype=torch.int64)
    labels = random_tensor * 255
    loss = F.cross_entropy(logits, labels)
    print(loss.item())
