import random

import torch
from torch.nn.functional import softmax

a = torch.zeros((1, 8, 192, 192), dtype=torch.float32)

for _ in range(32):
    a[0, random.randint(0, 7), random.randint(0, 191), random.randint(0, 191)] = 1

b = softmax(a.reshape(1, 8, -1), dim=2).reshape(1, 8, 192, 192).mean(dim=1).squeeze()

b = b * 192 * 192
print(b)
