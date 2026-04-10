from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 10000000000

def split_image(image_path, output_dir, block_size):
    """
    将大图切割成指定大小的小块

    参数:
    image_path: 原始图片路径
    output_dir: 输出小块图片的目录
    block_size: 小块图片的尺寸 (宽, 高)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开原始图片
    with Image.open(image_path) as img:
        # 获取原始图片尺寸
        width, height = img.size
        block_width, block_height = block_size

        # 计算可以切割成多少行和列
        rows = height // block_height
        cols = width // block_width

        print(f"原始图片尺寸: {width}x{height}")
        print(f"切割成 {rows}x{cols} 个 {block_width}x{block_height} 的小块")

        # 切割图片
        for i in range(rows):
            for j in range(cols):
                # 计算当前小块的坐标
                left = j * block_width
                upper = i * block_height
                right = left + block_width
                lower = upper + block_height

                # 裁剪小块
                block = img.crop((left, upper, right, lower))

                # 保存小块
                block.save(os.path.join(output_dir, f"block_5_9_{i}_{j}.png"))

        print(f"切割完成，共生成 {rows * cols} 个小块，保存在 {output_dir} 目录下")


if __name__ == "__main__":
    # 配置参数
    input_image = "D:/gaza/destory/ori/block_5_9.png"  # 5000x5000的原始图片路径
    output_directory = "D:\gaza\destory/p500"  # 输出目录
    block_dimensions = (500, 500)  # 小块尺寸

    # 执行切割
    split_image(input_image, output_directory, block_dimensions)
