import os
from PIL import Image
import torch
import torchvision.transforms as transforms


def size_change(folder_path ,target_size ,mode) :
    # 指定要处理的文件夹路径和目标尺寸



    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义图片转换操作
    resize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size)
    ])
    to_pil = transforms.ToPILImage()

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', 'tif')):  # 只处理图片文件
            file_path = os.path.join(folder_path, filename)

            # 打开图片并转换为张量
            image = Image.open(file_path).convert(mode)  # 保持 RGB 格式
            image_tensor = resize_transform(image).to(device)

            # 将张量转换回图片并保存（原地覆盖）
            modified_image = to_pil(image_tensor.cpu())
            modified_image.save(file_path)

    print("图片已修改为目标尺寸并保存！")

if __name__ == '__main__':
    size_change('D:\deep_learning\ISDNet-main\\root_path\imgs\\test',(1000,1000),'RGB')
    size_change('D:\deep_learning\ISDNet-main\\root_path\labels\\test',(1000,1000),'L')
