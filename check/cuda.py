import torch

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current Device Index: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Using CPU.")