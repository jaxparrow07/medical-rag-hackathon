# config.py - GPU-optimized settings

GPU_CONFIG = {
    'embedding_batch_size': 32,  # Adjust based on GPU memory
    'embedding_model': 'BAAI/bge-base-en-v1.5',  # 335MB model
    'torch_dtype': 'float16',  # Use half precision
    'max_seq_length': 512,
}

# In embedding.py, enable mixed precision
import torch

class OptimizedEmbedder(LocalEmbedder):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        if torch.cuda.is_available():
            # Enable mixed precision for faster inference
            self.model.half()
            torch.backends.cudnn.benchmark = True
