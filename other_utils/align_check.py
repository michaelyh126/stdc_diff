import os
def check_alignment(img_folder, label_folder):
    """
    检查 img_dir 和 rgb2id 目录下 train/val/test 文件是否对齐（忽略扩展名）
    """
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        img_subset_path = os.path.join(img_folder, subset)
        label_subset_path = os.path.join(label_folder, subset)

        img_files = sorted(os.listdir(img_subset_path))
        label_files = sorted(os.listdir(label_subset_path))

        # 统计文件数量
        if len(img_files) != len(label_files):
            print(f"[错误] {subset} 集图像数量与标签数量不匹配！")
            print(f"    图像文件数: {len(img_files)}, 标签文件数: {len(label_files)}")
            continue

        # 提取文件名（去掉扩展名）
        img_names = {os.path.splitext(f)[0] for f in img_files}
        label_names = {os.path.splitext(f)[0] for f in label_files}

        # 找出不匹配的项
        img_only = img_names - label_names
        label_only = label_names - img_names

        if img_only or label_only:
            print(f"[警告] {subset} 集存在不匹配的文件：")
            if img_only:
                print(f"    仅在图像目录中的文件: {sorted(img_only)}")
            if label_only:
                print(f"    仅在标签目录中的文件: {sorted(label_only)}")
        else:
            print(f"[✓] {subset} 集对齐正确！")

if __name__ == '__main__':
    # 定义路径
    img_folder = "/root/autodl-tmp/deepglobe1/land-train/img_dir"
    label_folder = "/root/autodl-tmp/deepglobe1/land-train/rgb2id"

    # 运行对齐检查
    check_alignment(img_folder, label_folder)
