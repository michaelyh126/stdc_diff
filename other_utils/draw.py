from PIL import Image, ImageDraw


def draw_red_rectangle(image_path, output_path, left, top, right, bottom):
    """
    在图像上指定位置绘制红色矩形框

    参数:
    image_path (str): 输入图像路径
    output_path (str): 输出图像路径
    left (int): 矩形左上角x坐标
    top (int): 矩形左上角y坐标
    right (int): 矩形右下角x坐标
    bottom (int): 矩形右下角y坐标
    """
    try:
        # 打开图像
        with Image.open(image_path) as img:
            # 创建绘图对象
            draw = ImageDraw.Draw(img)

            # 绘制红色矩形 (宽度为2像素)
            draw.rectangle([left, top, right, bottom], outline="green", width=30)

            # 保存结果
            img.save(output_path)
            print(f"已在位置 ({left}, {top}, {right}, {bottom}) 绘制红框")
            print(f"处理后的图像已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到图像文件 '{image_path}'")
    except Exception as e:
        print(f"错误: 处理图像时出错 - {e}")


if __name__ == "__main__":
    # 使用示例 - 请替换为实际的图像路径
    input_image = "D:/dataset/998002_sat_ours.jpg"
    output_image = "E:/实验图片/998002_sat_ours.jpg"

    # 指定矩形框的位置 (左, 上, 右, 下)
    rectangle_coords = (16, 8, 824, 1158)

    draw_red_rectangle(input_image, output_image, *rectangle_coords)
