import torch
import math
import os
import time
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import torch.nn as nn
import torch.nn.functional as F
# import cv2
import numpy as np
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .diff_fusion import FeatureFusionModule
from .harr import HarrUp
from other_utils.heatmap import save_image,save_heatmap,visualize_feature_map
from mmseg.ops import resize
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .pid import Bag,AdaptiveFrequencyFusion,AddFuse
from .diff_head import DiffHead
from .diff_point import DiffPoint
from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
from .spnetv2 import SpNetV2
from other_utils.split_tensor import split_tensor,restore_tensor
from .pid import segmenthead,CBAMLayer
from .sdd_stdc_head import ShallowNet
from .stdc_rep_head import ShallowNet_rep
from .stdc_lk_head import ShallowNet_lk,ShallowNet_lk2
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .isdhead import RelationAwareFusion
from mmseg.models.sampler.dysample import DySample
from other_utils.histogram import tensor_histogram
from mmseg.models.decode_heads.isdhead import SRDecoder
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from .stdc_rf import ShallowNet_rf63
from .pidnet_single import PIDNet
# from mmseg.models.decode_heads.UANet.models.UANet_O import UANet_Res50

# def _tensor_img_to_uint8(img_tensor,
#                          mean=(123.675, 116.28, 103.53),
#                          std=(58.395, 57.12, 57.375),
#                          to_bgr=True):
#     img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
#     mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
#     std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
#     img = img * std + mean
#     img = np.clip(img, 0, 255).astype(np.uint8)
#     if to_bgr:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     return img
#
#
# def _colorize_prob_map(prob_map):
#     vis = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
#     vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
#     return vis
#
#
# def _colorize_label_map(label_map):
#     label_map = label_map.astype(np.uint8)
#     if label_map.max() > 0:
#         vis = (label_map.astype(np.float32) / label_map.max() * 255).astype(np.uint8)
#     else:
#         vis = label_map.astype(np.uint8)
#     vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
#     return vis
#
#
# def _ensure_bgr_same_size(img, W, H, interp=cv2.INTER_NEAREST):
#     if img.ndim == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     if img.shape[0] != H or img.shape[1] != W:
#         img = cv2.resize(img, (W, H), interpolation=interp)
#     return img
#
#
#
#
#
# def visualize_refine_patch_debug(
#     inputs,
#     output,
#     save_dir="./refine_debug_v2",
#     target_class=1,
#     downsample_ratio=0.25,
#     patch_size_input=100,
#     min_area=0,
#     mean=(123.675, 116.28, 103.53),
#     std=(58.395, 57.12, 57.375),
# ):
#     """
#     这个版本同时可视化：
#     1) output/fusion 尺度下的 bbox / patch
#     2) 原图尺度下的 bbox_input / patch_input(固定100x100)
#
#     要求 get_refine_patches_from_output 已经换成新版，返回字段里包含：
#         bbox, bbox_input,
#         center, center_input,
#         orig_bbox, orig_bbox_input,
#         is_large, area_ds
#     """
#     os.makedirs(save_dir, exist_ok=True)
#
#     B, _, H_in, W_in = inputs.shape
#     _, C, H_out, W_out = output.shape
#
#     # 取得 patch 信息（新版）
#     patch_infos = get_refine_patches_from_output(
#         output=output,
#         input_shape=(H_in, W_in),
#         target_class=target_class,
#         downsample_ratio=downsample_ratio,
#         patch_size_input=patch_size_input,
#         min_area=min_area,
#     )
#
#     # 一些基础可视化
#     pred_label = torch.argmax(output, dim=1)                  # [B,H_out,W_out]
#     target_prob = torch.softmax(output, dim=1)[:, target_class]
#     fg_mask = (pred_label == target_class).float().unsqueeze(1)
#
#     ds_h = max(1, int(round(H_out * downsample_ratio)))
#     ds_w = max(1, int(round(W_out * downsample_ratio)))
#     fg_mask_ds = F.interpolate(fg_mask, size=(ds_h, ds_w), mode="nearest")
#
#     scale_out_to_in_x = W_in / W_out
#     scale_out_to_in_y = H_in / H_out
#
#     def add_title(im, title):
#         im = im.copy()
#         cv2.putText(
#             im, title, (10, 28),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#             (255, 255, 255), 2, cv2.LINE_AA
#         )
#         return im
#
#     def draw_center(im, cx, cy, color=(255, 0, 255), r=4):
#         cv2.circle(im, (int(cx), int(cy)), r, color, -1)
#
#     for b in range(B):
#         # -------------------------
#         # 原图空间图像
#         # -------------------------
#         img_in = _tensor_img_to_uint8(inputs[b], mean=mean, std=std, to_bgr=True)
#
#         # -------------------------
#         # output 空间图像
#         # -------------------------
#         img_out = cv2.resize(img_in, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
#
#         pred_np = pred_label[b].detach().cpu().numpy().astype(np.uint8)
#         prob_np = target_prob[b].detach().cpu().numpy().astype(np.float32)
#         fg_np = fg_mask[b, 0].detach().cpu().numpy().astype(np.uint8)
#         fg_ds_np = fg_mask_ds[b, 0].detach().cpu().numpy().astype(np.uint8)
#
#         pred_vis_out = _colorize_label_map(pred_np)
#         prob_vis_out = _colorize_prob_map(prob_np)
#
#         fg_vis_out = cv2.cvtColor((fg_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#
#         fg_ds_up_out = cv2.resize(
#             (fg_ds_np * 255).astype(np.uint8),
#             (W_out, H_out),
#             interpolation=cv2.INTER_NEAREST
#         )
#         fg_ds_vis_out = cv2.cvtColor(fg_ds_up_out, cv2.COLOR_GRAY2BGR)
#
#         # -------------------------
#         # output 空间：框可视化
#         # -------------------------
#         out_box_vis = img_out.copy()
#
#         # 下采样连通域先单独算一下，只为画 small / raw
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#             fg_ds_np, connectivity=8
#         )
#         scale_x = W_out / ds_w
#         scale_y = H_out / ds_h
#
#         raw_count = 0
#         small_count = 0
#         large_count = 0
#         kept_count = 0
#
#         for comp_id in range(1, num_labels):
#             x = int(stats[comp_id, cv2.CC_STAT_LEFT])
#             y = int(stats[comp_id, cv2.CC_STAT_TOP])
#             w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
#             h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
#             area = int(stats[comp_id, cv2.CC_STAT_AREA])
#
#             ox1 = int(round(x * scale_x))
#             oy1 = int(round(y * scale_y))
#             ox2 = int(round((x + w) * scale_x))
#             oy2 = int(round((y + h) * scale_y))
#
#             ox1 = max(0, min(ox1, W_out - 1))
#             oy1 = max(0, min(oy1, H_out - 1))
#             ox2 = max(0, min(ox2, W_out))
#             oy2 = max(0, min(oy2, H_out))
#
#             raw_count += 1
#
#             # 青色：所有 raw 连通域
#             cv2.rectangle(out_box_vis, (ox1, oy1), (ox2, oy2), (255, 255, 0), 1)
#
#             if area < min_area:
#                 small_count += 1
#                 # 红色：太小被过滤
#                 cv2.rectangle(out_box_vis, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
#
#         # 再画 patch_infos 里的保留/大目标
#         for info in patch_infos[b]:
#             if info["is_large"]:
#                 large_count += 1
#                 x1, y1, x2, y2 = info["orig_bbox"]
#                 cv2.rectangle(out_box_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             else:
#                 kept_count += 1
#                 ox1, oy1, ox2, oy2 = info["orig_bbox"]
#                 px1, py1, px2, py2 = info["bbox"]
#                 cx, cy = info["center"]
#
#                 # 绿框：output空间目标bbox
#                 cv2.rectangle(out_box_vis, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
#                 # 蓝框：output空间裁剪patch
#                 cv2.rectangle(out_box_vis, (px1, py1), (px2, py2), (255, 0, 0), 2)
#                 draw_center(out_box_vis, cx, cy)
#
#         # -------------------------
#         # 原图空间：框可视化
#         # -------------------------
#         in_box_vis = img_in.copy()
#         for info in patch_infos[b]:
#             if info["is_large"]:
#                 x1, y1, x2, y2 = info["orig_bbox_input"]
#                 cv2.rectangle(in_box_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             else:
#                 ox1, oy1, ox2, oy2 = info["orig_bbox_input"]
#                 px1, py1, px2, py2 = info["bbox_input"]
#                 cx, cy = info["center_input"]
#
#                 # 绿框：原图尺度目标bbox
#                 cv2.rectangle(in_box_vis, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
#                 # 蓝框：原图尺度固定100x100 patch
#                 cv2.rectangle(in_box_vis, (px1, py1), (px2, py2), (255, 0, 0), 2)
#                 draw_center(in_box_vis, cx, cy)
#
#         # -------------------------
#         # 为了拼图，把 output 那几张都 resize 到原图大小
#         # -------------------------
#         prob_show = _ensure_bgr_same_size(prob_vis_out, W_in, H_in, interp=cv2.INTER_NEAREST)
#         pred_show = _ensure_bgr_same_size(pred_vis_out, W_in, H_in, interp=cv2.INTER_NEAREST)
#         fg_show = _ensure_bgr_same_size(fg_vis_out, W_in, H_in, interp=cv2.INTER_NEAREST)
#         fg_ds_show = _ensure_bgr_same_size(fg_ds_vis_out, W_in, H_in, interp=cv2.INTER_NEAREST)
#         out_box_show = _ensure_bgr_same_size(out_box_vis, W_in, H_in, interp=cv2.INTER_NEAREST)
#
#         img_in_show = _ensure_bgr_same_size(img_in, W_in, H_in)
#         in_box_show = _ensure_bgr_same_size(in_box_vis, W_in, H_in)
#
#         # 再加标题
#         img_in_show = add_title(img_in_show, "input_image")
#         prob_show = add_title(prob_show, f"output_prob_cls{target_class}")
#         pred_show = add_title(pred_show, "output_pred_label")
#         fg_show = add_title(fg_show, "output_argmax_mask")
#         fg_ds_show = add_title(fg_ds_show, f"output_mask_downsample_x{downsample_ratio}")
#         out_box_show = add_title(out_box_show, "output_boxes: raw/cyan kept_blue+green large_yellow")
#
#         in_box_show = add_title(in_box_show, f"input_boxes: patch_blue={patch_size_input}x{patch_size_input}")
#
#         # 额外做一张：把 output 空间的 kept patch 映射回原图，只为了肉眼比对
#         remap_vis = img_in.copy()
#         for info in patch_infos[b]:
#             if info["is_large"]:
#                 continue
#             px1, py1, px2, py2 = info["bbox"]
#             rx1 = int(round(px1 * scale_out_to_in_x))
#             ry1 = int(round(py1 * scale_out_to_in_y))
#             rx2 = int(round(px2 * scale_out_to_in_x))
#             ry2 = int(round(py2 * scale_out_to_in_y))
#             cv2.rectangle(remap_vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)  # 青
#             bx1, by1, bx2, by2 = info["bbox_input"]
#             cv2.rectangle(remap_vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)     # 蓝
#         remap_vis = add_title(remap_vis, "compare: output_patch_remap_cyan vs input_patch_blue")
#
#         # 统计面板
#         stat_vis = np.zeros_like(img_in)
#         stat_lines = [
#             f"input  : {H_in} x {W_in}",
#             f"output : {H_out} x {W_out}",
#             f"target_class = {target_class}",
#             f"downsample_ratio = {downsample_ratio}",
#             f"patch_size_input = {patch_size_input}",
#             f"raw_cc   = {raw_count}",
#             f"small    = {small_count}",
#             f"large    = {large_count}",
#             f"kept     = {kept_count}",
#         ]
#         y0 = 40
#         for i, line in enumerate(stat_lines):
#             cv2.putText(
#                 stat_vis, line, (20, y0 + i * 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9,
#                 (255, 255, 255), 2, cv2.LINE_AA
#             )
#         stat_vis = add_title(stat_vis, "stats")
#
#         # 3 x 3 拼图
#         row1 = np.concatenate([img_in_show, prob_show, pred_show], axis=1)
#         row2 = np.concatenate([fg_show, fg_ds_show, out_box_show], axis=1)
#         row3 = np.concatenate([in_box_show, remap_vis, stat_vis], axis=1)
#         canvas = np.concatenate([row1, row2, row3], axis=0)
#
#         save_path = os.path.join(save_dir, f"debug_v2_{b:02d}.jpg")
#         cv2.imwrite(save_path, canvas)
#
#         print(
#             f"[debug_v2 {b}] "
#             f"input=({H_in},{W_in}) output=({H_out},{W_out}) "
#             f"raw_cc={raw_count} small={small_count} large={large_count} kept={kept_count} "
#             f"saved={save_path}"
#         )
#
#     return patch_infos
# def get_refine_patches_from_output(
#     output,
#     input_shape,              # (H_in, W_in)
#     target_class=1,
#     downsample_ratio=0.25,
#     patch_size_input=100,     # 你真正想要的原图 patch 尺寸
#     min_area=0,
# ):
#     """
#     Args:
#         output: [B, C, H, W]，分割 logits（output/fusion 尺度）
#         input_shape: (H_in, W_in)，原图尺度
#         target_class: 目标类别
#         downsample_ratio: 连通域检测前，对 output mask 再降采样的比例
#         patch_size_input: 原图空间里最终 patch 的尺寸（固定 100）
#         min_area: 在 downsample 后 mask 上允许的最小连通域面积
#
#     Returns:
#         batch_patches: list[list[dict]]
#             每个 dict:
#             {
#                 "bbox": [x1, y1, x2, y2],              # output/fusion 空间裁剪框
#                 "bbox_input": [ix1, iy1, ix2, iy2],   # 原图空间固定100x100 patch
#                 "center": [cx_out, cy_out],           # output 空间中心
#                 "center_input": [cx_in, cy_in],       # 原图空间中心
#                 "orig_bbox": [ox1, oy1, ox2, oy2],    # output 空间目标 bbox
#                 "orig_bbox_input": [ix1, iy1, ix2, iy2], # 原图空间目标 bbox
#                 "is_large": bool,
#                 "area_ds": int
#             }
#     """
#     assert output.dim() == 4, "output must be [B, C, H, W]"
#
#     B, C, H, W = output.shape
#     H_in, W_in = input_shape
#
#     # 原图 / output 的尺度比
#     scale_in_x = W_in / W
#     scale_in_y = H_in / H
#
#     # 原图100x100对应到 output/fusion 上的固定 patch 尺寸
#     patch_w_out = max(1, int(round(patch_size_input / scale_in_x)))
#     patch_h_out = max(1, int(round(patch_size_input / scale_in_y)))
#     half_w_out = patch_w_out // 2
#     half_h_out = patch_h_out // 2
#
#     pred_label = torch.argmax(output, dim=1)                  # [B,H,W]
#     fg_mask = (pred_label == target_class).float().unsqueeze(1)
#
#     ds_h = max(1, int(round(H * downsample_ratio)))
#     ds_w = max(1, int(round(W * downsample_ratio)))
#     fg_mask_ds = F.interpolate(fg_mask, size=(ds_h, ds_w), mode="nearest")
#
#     # ds 空间 -> output 空间
#     scale_x = W / ds_w
#     scale_y = H / ds_h
#
#     batch_patches = []
#
#     for b in range(B):
#         mask_np = fg_mask_ds[b, 0].detach().cpu().numpy().astype(np.uint8)
#
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#             mask_np, connectivity=8
#         )
#
#         patches_this_img = []
#
#         for comp_id in range(1, num_labels):
#             x = int(stats[comp_id, cv2.CC_STAT_LEFT])
#             y = int(stats[comp_id, cv2.CC_STAT_TOP])
#             w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
#             h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
#             area = int(stats[comp_id, cv2.CC_STAT_AREA])
#
#             # 先别筛太狠
#             if area < min_area:
#                 continue
#
#             # 1) ds 空间 bbox -> output 空间 bbox
#             ox1 = int(round(x * scale_x))
#             oy1 = int(round(y * scale_y))
#             ox2 = int(round((x + w) * scale_x))
#             oy2 = int(round((y + h) * scale_y))
#
#             ox1 = max(0, min(ox1, W - 1))
#             oy1 = max(0, min(oy1, H - 1))
#             ox2 = max(0, min(ox2, W))
#             oy2 = max(0, min(oy2, H))
#
#             # 2) output 空间 bbox -> 原图空间 bbox
#             ix1 = int(round(ox1 * scale_in_x))
#             iy1 = int(round(oy1 * scale_in_y))
#             ix2 = int(round(ox2 * scale_in_x))
#             iy2 = int(round(oy2 * scale_in_y))
#
#             ix1 = max(0, min(ix1, W_in - 1))
#             iy1 = max(0, min(iy1, H_in - 1))
#             ix2 = max(0, min(ix2, W_in))
#             iy2 = max(0, min(iy2, H_in))
#
#             iw_in = ix2 - ix1
#             ih_in = iy2 - iy1
#
#             # 3) 大目标判断必须在原图尺度下做
#             is_large = (iw_in > patch_size_input) or (ih_in > patch_size_input)
#
#             if is_large:
#                 patches_this_img.append({
#                     "bbox": None,
#                     "bbox_input": None,
#                     "center": None,
#                     "center_input": None,
#                     "orig_bbox": [ox1, oy1, ox2, oy2],
#                     "orig_bbox_input": [ix1, iy1, ix2, iy2],
#                     "is_large": True,
#                     "area_ds": area,
#                 })
#                 continue
#
#             # 4) 原图空间中心
#             cx_in = int(round((ix1 + ix2) / 2.0))
#             cy_in = int(round((iy1 + iy2) / 2.0))
#
#             # 5) 先在原图空间生成严格 100x100 patch
#             half_in = patch_size_input // 2
#             ipx1 = cx_in - half_in
#             ipy1 = cy_in - half_in
#             ipx2 = ipx1 + patch_size_input
#             ipy2 = ipy1 + patch_size_input
#
#             # 原图边界修正，保证 bbox_input 一定是100x100
#             if ipx1 < 0:
#                 ipx2 -= ipx1
#                 ipx1 = 0
#             if ipy1 < 0:
#                 ipy2 -= ipy1
#                 ipy1 = 0
#             if ipx2 > W_in:
#                 ipx1 -= (ipx2 - W_in)
#                 ipx2 = W_in
#             if ipy2 > H_in:
#                 ipy1 -= (ipy2 - H_in)
#                 ipy2 = H_in
#
#             ipx1 = max(0, ipx1)
#             ipy1 = max(0, ipy1)
#             ipx2 = min(W_in, ipx2)
#             ipy2 = min(H_in, ipy2)
#
#             # 理论上这里应始终保持100x100；极小图像除外
#             if (ipx2 - ipx1) != patch_size_input or (ipy2 - ipy1) != patch_size_input:
#                 continue
#
#             # 6) 再生成 output/fusion 空间的固定裁剪框
#             cx_out = int(round(cx_in / scale_in_x))
#             cy_out = int(round(cy_in / scale_in_y))
#
#             px1 = cx_out - half_w_out
#             py1 = cy_out - half_h_out
#             px2 = px1 + patch_w_out
#             py2 = py1 + patch_h_out
#
#             if px1 < 0:
#                 px2 -= px1
#                 px1 = 0
#             if py1 < 0:
#                 py2 -= py1
#                 py1 = 0
#             if px2 > W:
#                 px1 -= (px2 - W)
#                 px2 = W
#             if py2 > H:
#                 py1 -= (py2 - H)
#                 py2 = H
#
#             px1 = max(0, px1)
#             py1 = max(0, py1)
#             px2 = min(W, px2)
#             py2 = min(H, py2)
#
#             if (px2 - px1) != patch_w_out or (py2 - py1) != patch_h_out:
#                 continue
#
#             patches_this_img.append({
#                 "bbox": [px1, py1, px2, py2],              # 给 fusion/output 裁剪用
#                 "bbox_input": [ipx1, ipy1, ipx2, ipy2],   # 原图上真正的100x100
#                 "center": [cx_out, cy_out],
#                 "center_input": [cx_in, cy_in],
#                 "orig_bbox": [ox1, oy1, ox2, oy2],
#                 "orig_bbox_input": [ix1, iy1, ix2, iy2],
#                 "is_large": False,
#                 "area_ds": area,
#             })
#
#         batch_patches.append(patches_this_img)
#
#     return batch_patches
#
# def crop_patches(
#     fusion,
#     inputs,
#     patch_infos,
#     fusion_interp_mode="bilinear",
#     align_corners=False,
# ):
#     """
#     将 fusion 上采样到原图大小后，与原图 input 在通道维拼接，
#     然后按照 patch_infos 里的 bbox_input 裁 patch。
#
#     Args:
#         fusion: [B, C_f, H_f, W_f]
#         inputs: [B, C_in, H_in, W_in]
#         patch_infos: get_refine_patches_from_output 新版返回结果
#                      其中每个小目标应包含:
#                         info["bbox_input"] = [x1, y1, x2, y2]
#         fusion_interp_mode: 上采样模式，默认 bilinear
#
#     Returns:
#         patch_tensors: list of Tensor, each [1, C_f + C_in, patch_h, patch_w]
#         patch_meta: list of (b, info)
#         fusion_cat: [B, C_f + C_in, H_in, W_in]，可选返回便于调试
#     """
#     assert fusion.dim() == 4, "fusion must be [B, C, H, W]"
#     assert inputs.dim() == 4, "inputs must be [B, C, H, W]"
#     assert fusion.size(0) == inputs.size(0), "fusion and inputs batch size must match"
#
#     B, C_f, H_f, W_f = fusion.shape
#     B2, C_in, H_in, W_in = inputs.shape
#     assert B == B2
#
#     # 1) fusion 上采样到原图大小
#     fusion_up = F.interpolate(
#         fusion,
#         size=(H_in, W_in),
#         mode=fusion_interp_mode,
#         align_corners=align_corners if fusion_interp_mode in ["linear", "bilinear", "bicubic", "trilinear"] else None,
#     )
#
#     # 2) 与原图拼接
#     fusion_cat = torch.cat([fusion_up, inputs], dim=1)   # [B, C_f + C_in, H_in, W_in]
#
#     patch_tensors = []
#     patch_meta = []
#
#     # 3) 按原图坐标 bbox_input 裁 patch
#     for b in range(B):
#         for info in patch_infos[b]:
#             if info["is_large"]:
#                 continue
#
#             if "bbox_input" not in info or info["bbox_input"] is None:
#                 continue
#
#             x1, y1, x2, y2 = info["bbox_input"]
#
#             # 边界保护
#             x1 = max(0, min(int(x1), W_in - 1))
#             y1 = max(0, min(int(y1), H_in - 1))
#             x2 = max(0, min(int(x2), W_in))
#             y2 = max(0, min(int(y2), H_in))
#
#             if x2 <= x1 or y2 <= y1:
#                 continue
#
#             patch = fusion_cat[b:b+1, :, y1:y2, x1:x2]   # [1, C_f + C_in, ph, pw]
#             patch_tensors.append(patch)
#             patch_meta.append((b, info))
#
#     return patch_tensors, patch_meta, fusion_cat
#
#
# def paste_patch_logits_to_fullres(
#     patch_logits,      # [N, num_classes, patch_h, patch_w]
#     patch_meta,        # list of (b, info)
#     batch_size,
#     num_classes,
#     image_size,        # (H_in, W_in)
# ):
#     """
#     将 patch 级别的 refine logits 还原回原图大小
#
#     Args:
#         patch_logits: [N, num_classes, ph, pw]，比如 f1
#         patch_meta: crop_patches 返回的 meta，长度为 N
#         batch_size: 原始 batch 大小 B
#         num_classes: 类别数
#         image_size: (H_in, W_in)
#
#     Returns:
#         full_logits: [B, num_classes, H_in, W_in]
#         valid_map:   [B, 1, H_in, W_in]，表示哪些区域被 patch 覆盖
#     """
#     H_in, W_in = image_size
#     device = patch_logits.device
#     dtype = patch_logits.dtype
#
#     full_logits = torch.zeros(
#         (batch_size, num_classes, H_in, W_in),
#         device=device,
#         dtype=dtype
#     )
#     count_map = torch.zeros(
#         (batch_size, 1, H_in, W_in),
#         device=device,
#         dtype=dtype
#     )
#
#     for i, (b, info) in enumerate(patch_meta):
#         if "bbox_input" not in info or info["bbox_input"] is None:
#             continue
#
#         x1, y1, x2, y2 = info["bbox_input"]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#         if x2 <= x1 or y2 <= y1:
#             continue
#
#         patch_logit = patch_logits[i:i+1]   # [1, num_classes, ph, pw]
#         target_h = y2 - y1
#         target_w = x2 - x1
#
#         # 防止 patch_logit 尺寸和 bbox_input 不一致
#         if patch_logit.shape[-2] != target_h or patch_logit.shape[-1] != target_w:
#             patch_logit = F.interpolate(
#                 patch_logit,
#                 size=(target_h, target_w),
#                 mode="bilinear",
#                 align_corners=False
#             )
#
#         full_logits[b:b+1, :, y1:y2, x1:x2] += patch_logit
#         count_map[b:b+1, :, y1:y2, x1:x2] += 1.0
#
#     valid_map = (count_map > 0).float()
#
#     # 对重叠 patch 做平均
#     full_logits = full_logits / count_map.clamp(min=1.0)
#
#     return full_logits, valid_map



def split_to_patches(inputs, patch_size):
    """
    inputs: [B, C, H, W]
    patch_size: x

    return:
        patches: [B * n_h * n_w, C, x, x]
        meta: dict, 用于恢复
    """
    B, C, H, W = inputs.shape
    x = patch_size

    assert H % x == 0 and W % x == 0, \
        f"H and W must be divisible by patch_size, got H={H}, W={W}, patch_size={x}"

    n_h = H // x
    n_w = W // x

    # [B, C, H, W] -> [B, C, n_h, x, n_w, x]
    patches = inputs.reshape(B, C, n_h, x, n_w, x)

    # -> [B, n_h, n_w, C, x, x]
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()

    # -> [B*n_h*n_w, C, x, x]
    patches = patches.reshape(B * n_h * n_w, C, x, x)

    meta = {
        'B': B,
        'H': H,
        'W': W,
        'patch_size': x,
        'n_h': n_h,
        'n_w': n_w,
    }

    return patches, meta


def merge_patches(outputs, meta, out_size=None):
    """
    outputs: [B * n_h * n_w, C, h_p, w_p]
    meta: split_to_patches 返回的 meta
    out_size: 最终想恢复到的尺寸 (H, W)，默认用原图尺寸

    return:
        merged: [B, C, H_out, W_out]
    """
    B = meta['B']
    n_h = meta['n_h']
    n_w = meta['n_w']
    H = meta['H']
    W = meta['W']

    BN, C, h_p, w_p = outputs.shape
    assert BN == B * n_h * n_w, \
        f"Batch size mismatch: got {BN}, expected {B*n_h*n_w}"

    # [B*n_h*n_w, C, h_p, w_p]
    # -> [B, n_h, n_w, C, h_p, w_p]
    outputs = outputs.reshape(B, n_h, n_w, C, h_p, w_p)

    # -> [B, C, n_h, h_p, n_w, w_p]
    outputs = outputs.permute(0, 3, 1, 4, 2, 5).contiguous()

    # -> [B, C, n_h*h_p, n_w*w_p]
    merged = outputs.reshape(B, C, n_h * h_p, n_w * w_p)

    if out_size is None:
        out_size = (H, W)

    return merged

@HEADS.register_module()
class OMRFHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='pid', **kwargs):
        super(OMRFHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.patch_size=img_size[0]
        # self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
        #         type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        # self.stdc_net = ShallowNet_rf63(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        # self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.stdc_net = ShallowNet_lk(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_rep(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.fuse=AddFuse(self.channels,self.channels)
        self.cls_seg8 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.pid=PIDNet(m=2, n=3, num_classes=self.num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
        # self.convert_f8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        # self.convert_f16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        # self.fuse2=AddFuse(self.channels,self.channels)
        # self.cls_seg2 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        # self.stdc_net2 = ShallowNet_lk2(in_channels=3,num_classes=self.num_classes)
        # self.fusion_reduce_for_patch = nn.Conv2d(self.channels, 61, kernel_size=1, stride=1, padding=0)
        # self.uanet = UANet_Res50(32, 2)



    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        ori_size = inputs.shape[-2:]  # (H, W)
        inputs, patch_meta = split_to_patches(inputs, self.patch_size)
        shallow_feat8, shallow_feat16 = self.stdc_net(inputs)

        # add fusion
        shallow_feat16 = self.convert_shallow16(shallow_feat16)
        shallow_feat8=self.convert_shallow8(shallow_feat8)
        _, _, h, w = shallow_feat8.size()
        shallow_feat16 = F.interpolate(shallow_feat16, size=(h , w ), mode='bilinear', align_corners=False)
        fusion=self.addConv(shallow_feat8,shallow_feat16)
        fusion = merge_patches(fusion, patch_meta, out_size=ori_size)

        if self.decoder_flag=='pid':
            if train_flag==True:
                output = self.cls_seg(fusion)
                predict = self.pid.forward_dual(fusion)
                aux_output = self.cls_seg8(shallow_feat8)
                aux_output = merge_patches(aux_output, patch_meta, out_size=ori_size)
                return output,aux_output,predict
            else:
                predict = self.pid.forward_dual(fusion)
                return predict
        if self.decoder_flag=='ori':
            if train_flag==True:
                output = self.cls_seg(fusion)
                aux_output = self.cls_seg8(shallow_feat8)
                aux_output = merge_patches(aux_output, patch_meta, out_size=ori_size)
                return output,aux_output
            else:
                output = self.cls_seg(fusion)
                return output


    # def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
    #     """Forward function."""
    #     # if train_flag==False:
    #     #     self.stdc_net.switch_to_deploy()
    #
    #     shallow_feat8, shallow_feat16 = self.stdc_net(inputs)
    #
    #     # add fusion
    #     shallow_feat16 = self.convert_shallow16(shallow_feat16)
    #     shallow_feat8=self.convert_shallow8(shallow_feat8)
    #     _, _, h, w = shallow_feat8.size()
    #     shallow_feat16 = F.interpolate(shallow_feat16, size=(h , w ), mode='bilinear', align_corners=False)
    #     fusion=self.addConv(shallow_feat8,shallow_feat16)
    #     output = self.cls_seg(fusion)
    #     # visualize_refine_patch_debug(
    #     #     inputs=inputs,
    #     #     output=output,
    #     #     save_dir="/root/autodl-tmp/ISDNet-main/refine_debug",
    #     #     target_class=1,
    #     #     downsample_ratio=0.25,
    #     #     patch_size_input=100,
    #     #     min_area=0,
    #     # )
    #     patch_infos=get_refine_patches_from_output(output,inputs.shape[-2:])
    #
    #     fusion_d=self.fusion_reduce_for_patch(fusion)
    #     patch_tensors, patch_meta, fusion_cat = crop_patches(
    #         fusion=fusion_d,
    #         inputs=inputs,
    #         patch_infos=patch_infos,
    #     )
    #     patch_batch = torch.cat(patch_tensors, dim=0)
    #
    #
    #     if len(patch_tensors) > 0:
    #         patch_batch = torch.cat(patch_tensors, dim=0)  # [N, 64, 100, 100]
    #         # f5, f4, f3, f2, f1 = self.uanet(patch_batch)
    #         f8, f16 = self.stdc_net2(patch_batch)
    #         f8=self.convert_f8(f8)
    #         f16=self.convert_f16(f16)
    #         f16 = F.interpolate(f16, size=f8.shape[-2:], mode='bilinear', align_corners=False)
    #         fusion2=self.fuse2(f8,f16)
    #         f1=self.cls_seg2(fusion2)
    #
    #
    #         refine_logits_full, refine_valid_map = paste_patch_logits_to_fullres(
    #             patch_logits=f1,
    #             patch_meta=patch_meta,
    #             batch_size=inputs.size(0),
    #             num_classes=self.num_classes,
    #             image_size=inputs.shape[-2:],
    #         )
    #     else:
    #         refine_logits_full = None
    #         refine_valid_map = None
    #
    #     main_logits_up = F.interpolate(
    #         output,
    #         size=inputs.shape[-2:],
    #         mode='bilinear',
    #         align_corners=False
    #     )
    #
    #     if refine_logits_full is not None:
    #         final_logits = main_logits_up * (1 - refine_valid_map) + refine_logits_full * refine_valid_map
    #     else:
    #         final_logits = main_logits_up
    #
    #     if train_flag==True:
    #
    #         aux_output = self.cls_seg8(shallow_feat8)
    #         return output,aux_output,f1,final_logits
    #     else:
    #         return output







    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        if self.decoder_flag=='ori':
            output,aux_output= self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses= self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            return  losses,losses_aux
        elif self.decoder_flag=='pid':
            output,aux_output,predict= self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses = self.losses(predict, gt_semantic_seg)
            losses_middle= self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            return  losses,losses_middle,losses_aux
        else:
            output,aux_output,f1,final_logits = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses = self.losses(final_logits, gt_semantic_seg)
            losses_s = self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            # losses_f1 = self.losses(f1, gt_semantic_seg)
            # losses_f2 = self.losses(f2, gt_semantic_seg)
            # losses_f3 = self.losses(f3, gt_semantic_seg)
            # losses_f4 = self.losses(f4, gt_semantic_seg)
            # losses_f5 = self.losses(f5, gt_semantic_seg)
            return  losses,losses_s,losses_aux



    def forward_test(self, inputs, prev_output, img_metas, test_cfg,mask=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        return self.forward(inputs, prev_output, False,diff_pred_deep=mask)

