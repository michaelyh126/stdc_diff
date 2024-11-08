import gc


def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    import torch
    import mmcv

    print(torch.__version__)
    #cuda是否可用；
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    report_gpu()


