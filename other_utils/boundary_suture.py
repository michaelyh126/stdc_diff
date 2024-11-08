import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import math


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_index_matrix(b, h, w, topk_indices):
    # 创建一个全零矩阵
    index_matrix = torch.zeros(b, h, w,device=topk_indices.device)
    row_indices = topk_indices // w
    col_indices = topk_indices % w
    # 利用高级索引设置矩阵中的值
    for i in range(b):
        index_matrix[i, row_indices[i], col_indices[i]] = 1


    # # 将展平的索引转换回二维索引
    # for i in range(b):
    #     for idx in topk_indices[i]:
    #         row = idx // w
    #         col = idx % w
    #         index_matrix[i,row, col] = 1

    return index_matrix


def create_mask(input,block_height=256,block_width=256,margin_size=10):
    batch_size,channels,height,width=input.shape
    mask=torch.zeros(batch_size,channels,height,width)
    column_num = math.floor(width/block_width) if width%block_width!=0 else int(width/block_width-1)
    row_num = math.floor(height/block_height) if height%block_height!=0 else int(height/block_height-1)

    for i in range(column_num):
        t=i+1
        mask[:,:,:,block_width*t-margin_size:block_width*t+margin_size]=1
    for j in range(row_num):
        t=j+1
        mask[:,:,block_height*t-margin_size:block_height*t+margin_size,:]=1
    return mask


def calculate_uncertainty(seg_logits):
    top2_scores = torch.topk(seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)


def extract_uncertainty(input,uncertainty_func,num_points):
    batch_size, channels, height, width = input.shape
    uncertainty_map=uncertainty_func(input)
    batch_size, _, height, width = uncertainty_map.shape
    h_step = 1.0 / height
    w_step = 1.0 / width

    uncertainty_map = uncertainty_map.view(batch_size, height * width)
    num_points = min(height * width, num_points)

    #忽略非边界（值为0）对topk的影响
    small_value = torch.tensor(float('-inf')).to(device)
    tensor_no_zero = torch.where(uncertainty_map == 0, small_value, uncertainty_map).to(device)

    point_indices = tensor_no_zero.topk(num_points, dim=1)[1]
    uncertainty_map=create_index_matrix(height,width,point_indices)
    # point_coords = torch.zeros(
    #     batch_size,
    #     num_points,
    #     2,
    #     dtype=torch.float,
    #     device=input.device)
    # point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
    #                                         width).float() * w_step
    # point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
    #                                         width).float() * h_step
    return uncertainty_map


def find_max_density_area(input,height=1500,width=1500, block_height=256,block_width=256,window_size=100):
    column_num = math.floor(width/block_width) if width%block_width!=0 else int(width/block_width-1)
    row_num = math.floor(height/block_height) if height%block_height!=0 else int(height/block_height-1)
    dic={}
    i=0
    half_window_size=window_size//2
    while i<=row_num:
        i=i+1
        j=1
        while j*half_window_size+half_window_size<=width:
            window=input[i*block_height-half_window_size:i*block_height+half_window_size,j*half_window_size-half_window_size:j*half_window_size+half_window_size]
            count_of_uncertainty = (window == 1).sum().item()
            dic[(i*block_height,j*half_window_size)]=count_of_uncertainty
            j = j + 1

    m=0
    while m<=column_num:
        m=m+1
        n=1
        while n*half_window_size+half_window_size<=height:
            window=input[n*half_window_size-half_window_size:n*half_window_size+half_window_size,m*block_width-half_window_size:m*block_width+half_window_size]
            count_of_uncertainty = (window == 1).sum().item()
            dic[(n*half_window_size,m*block_width)] = count_of_uncertainty
            n = n + 1

    max_key = max(dic, key=dic.get)
    # max_value = my_dict[max_key]
    return max_key

class BoundarySuture(nn.Module):
    def init(self):
        super(BoundarySuture, self).init()

    def forward(self, input):
        return


# if __name__ == '__main__':
#     input=np.full((1,2,100,100),1)
#     mask=create_mask(input,10,10,1)
#     margin=input*mask
#     pass
