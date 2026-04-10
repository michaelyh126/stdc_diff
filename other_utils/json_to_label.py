import json
import os
from PIL import Image, ImageDraw


def create_mask_from_isat_json(json_path, output_mask_path, label_to_color,width,height):
    """
    根据单个 ISAT JSON 文件生成语义分割掩码图片。

    :param json_path: ISAT JSON 文件的路径。
    :param output_mask_path: 生成的掩码图片的保存路径。
    :param label_to_color: 一个字典，用于将类别标签映射到RGB颜色。
                           例如: {'person': (255, 0, 0), 'car': (0, 255, 0)}
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {json_path} 不是有效的JSON格式。")
        return





    # 创建一个与原图同样大小的空白RGB图像（黑色背景）
    # mask = Image.new('RGB', (width, height), (0, 0, 0))
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # 遍历所有标注对象
    for obj in data.get('objects', []):
        # label = obj['label']
        points = obj['segmentation']

        # 从映射中获取颜色，如果找不到则使用默认的白色
        color = 1

        # ISAT的points格式是 [[x1, y1], [x2, y2], ...]
        # PIL的ImageDraw.polygon需要 [(x1, y1), (x2, y2), ...] 格式
        # 所以需要转换一下
        polygon_points = [(p[0], p[1]) for p in points]

        # 绘制填充的多边形
        if len(polygon_points) > 2:  # 确保至少有3个点才能构成多边形
            draw.polygon(polygon_points, fill=color)

    # 保存生成的掩码图片
    mask.save(output_mask_path)
    print(f"成功生成掩码: {output_mask_path}")


def batch_convert_isat_to_masks(json_dir, mask_dir, label_to_color,width,height):
    """
    批量转换一个文件夹下的所有 ISAT JSON 文件。

    :param json_dir: 存放JSON文件的文件夹路径。
    :param mask_dir: 存放生成掩码图片的文件夹路径。
    :param label_to_color: 类别到颜色的映射字典。
    """
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        print(f"创建输出目录: {mask_dir}")

    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            # 生成与json同名的png图片
            mask_filename = os.path.splitext(filename)[0] + '.png'
            output_mask_path = os.path.join(mask_dir, mask_filename)

            create_mask_from_isat_json(json_path, output_mask_path, label_to_color,width,height)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 定义你的类别到颜色的映射 (这是最重要的部分，需要根据你的数据集自定义！)
    #    RGB颜色值，每个分量在0-255之间
    LABEL_TO_COLOR = {
        "ruins": (255, 0, 0),  # 红色
        # "car": (0, 255, 0),  # 绿色
        # "building": (0, 0, 255),  # 蓝色
        # "sky": (135, 206, 235),  # 天蓝色
        # "tree": (0, 128, 0),  # 深绿色
        # # 在这里添加你所有的类别...
    }

    # 2. 设置输入和输出文件夹路径
    JSON_FILES_DIRECTORY = 'D:\gaza\destory\P500_label'  # <-- 修改为你的JSON文件夹路径
    MASK_OUTPUT_DIRECTORY = 'D:\gaza\destory\p500_mask'  # <-- 修改为你想保存掩码的文件夹路径

    # 3. 执行批量转换
    batch_convert_isat_to_masks(JSON_FILES_DIRECTORY, MASK_OUTPUT_DIRECTORY, LABEL_TO_COLOR,500,500)

    # 如果只想转换单个文件，可以这样调用：
    # single_json_path = 'path/to/your/single_annotation.json'
    # single_mask_path = 'path/to/your/output_mask.png'
    # create_mask_from_isat_json(single_json_path, single_mask_path, LABEL_TO_COLOR)
