import os


def get_relative_file_paths(root_dir):
    # 定义允许的文件扩展名
    ALLOWED_EXTENSIONS = {'.js', '.vue', '.ts'}

    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否符合要求
            ext = os.path.splitext(filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                # 生成相对路径，并且不包含根目录
                rel_dir = os.path.relpath(dirpath, root_dir)
                if rel_dir == '.':
                    file_paths.append(filename)  # 处理根目录下的文件
                else:
                    file_paths.append(os.path.join(rel_dir, filename))
    return file_paths


if __name__ == '__main__':

    # 使用示例
    root_directory = "D:\\underwater\\under-water\\src"  # 替换成实际的文件夹路径
    relative_paths = get_relative_file_paths(root_directory)
    c=0
    for path in relative_paths:
        c=c+1
        print(path)
        print(c)
