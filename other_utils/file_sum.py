import os

def count_files_in_directory(directory):
    try:
        # 列出目录中的所有文件和文件夹
        files = os.listdir(directory)
        # 过滤出文件
        file_count = sum(1 for f in files if os.path.isfile(os.path.join(directory, f)))
        return file_count
    except Exception as e:
        print(f"发生错误: {e}")
        return None

if __name__ == '__main__':
    directory_path = 'D:\dataset\\aerial\imgs\\test'  # 替换为你要统计的文件夹路径
    file_count = count_files_in_directory(directory_path)
    print(f"文件夹中有 {file_count} 个文件。")
