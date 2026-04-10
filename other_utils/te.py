import torch
if __name__ == '__main__':

    print(f"PyTorch version: {torch.__version__}")

    # 检查 PyTorch 是否编译了 CUDA 支持
    if torch.cuda.is_available():
        print(f"CUDA is available. PyTorch was built with CUDA {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. This might be a CPU-only build of PyTorch.")
