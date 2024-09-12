import numpy as np
import lzma
import torch
import torch.nn as nn

# Exemplo de tensor NumPy
pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
tensor = torch.randn(2, 320, 64, 64, dtype=torch.float16)
reduced_tensor = pooling_layer(tensor)

# Salvar e comprimir com LZMA
with lzma.open('meu_tensor_comprimido_lzma2.npy.xz', 'wb') as f:
    np.save(f, reduced_tensor)

# Carregar o tensor comprimido com LZMA
with lzma.open('meu_tensor_comprimido_lzma2.npy.xz', 'rb') as f:
    tensor_carregado = np.load(f)



# Function to quantize a tensor
def quantize_tensor(tensor, num_bits=8):
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (2**num_bits - 1)
    zero_point = torch.round(-min_val / scale).long()
    
    quantized_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 2**num_bits - 1).long()
    
    return quantized_tensor, scale, zero_point

# Function to dequantize a tensor
def dequantize_tensor(quantized_tensor, scale, zero_point):
    return (quantized_tensor - zero_point) * scale

# Exemplo de tensor NumPy
tensor = torch.randn(2, 320, 64, 64, dtype=torch.float16)

# Quantize the tensor
quantized_tensor, scale, zero_point = quantize_tensor(tensor)

# Save and compress with LZMA
with lzma.open('meu_tensor_comprimido_lzmaquant.npy.xz', 'wb') as f:
    np.save(f, quantized_tensor.cpu().numpy())
    np.save(f, scale.item())  # Save scale
    np.save(f, zero_point.item())  # Save zero_point

# Load and decompress tensor with LZMA
with lzma.open('meu_tensor_comprimido_lzmaquant.npy.xz', 'rb') as f:
    quantized_tensor = np.load(f)
    scale = np.load(f)
    zero_point = np.load(f)

# Convert loaded values back to tensors
quantized_tensor = torch.tensor(quantized_tensor, dtype=torch.long)
scale = torch.tensor(scale, dtype=torch.float32)
zero_point = torch.tensor(zero_point, dtype=torch.long)

# Dequantize the tensor
dequantized_tensor = dequantize_tensor(quantized_tensor, scale, zero_point)