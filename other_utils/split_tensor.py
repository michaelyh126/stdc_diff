import torch

def split_tensor(tensor,scale=2):

    # 假设原始 tensor 为 bchw
    b, c, h, w = tensor.shape  # 示例形状


    # 将 tensor 切割成四个 (b, c, h//2, w//2) 的张量
    top_left = tensor[:, :, :h//scale, :w//scale]
    top_right = tensor[:, :, :h//scale, w//scale:]
    bottom_left = tensor[:, :, h//scale:, :w//scale]
    bottom_right = tensor[:, :, h//scale:, w//scale:]
    return top_left,top_right,bottom_left,bottom_right

def restore_tensor(top_left,top_right,bottom_left,bottom_right):
    # 如果需要将四个张量还原回原始的 bchw 形状
    top = torch.cat((top_left, top_right), dim=3)  # 在宽度方向拼接
    bottom = torch.cat((bottom_left, bottom_right), dim=3)  # 在宽度方向拼接
    restored_tensor = torch.cat((top, bottom), dim=2)  # 在高度方向拼接
    return restored_tensor

if __name__ == '__main__':
    tensor = torch.randn(1, 3, 5, 5)
    top_left,top_right,bottom_left,bottom_right=split_tensor(tensor,2)
    restored_tensor=restore_tensor(top_left,top_right,bottom_left,bottom_right)
    print('end')
