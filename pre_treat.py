import numpy as np
import torch

def cutmix(images, labels, beta):
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    lam = np.random.beta(beta, beta)
    lam = max(lam, 1 - lam)

    # 生成替换区域
    cx = np.random.uniform(0, images.size(2))
    cy = np.random.uniform(0, images.size(3))
    w = images.size(2) * np.sqrt(1 - lam)
    h = images.size(3) * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, images.size(2))))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, images.size(3))))

    mixed_images = images.clone()
    mixed_images[:, :, x0:x1, y0:y1] = shuffled_images[:, :, x0:x1, y0:y1]

    # 返回增强图像，对应标签及混合比例
    return mixed_images, labels, shuffled_labels, lam

def cutout(images, length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = images.size(0)
    h = images.size(2)
    w = images.size(3)

    # 生成遮挡区域
    mask = np.ones((batch_size, h, w), np.float32)
    for i in range(batch_size):
        x = np.random.randint(w)
        y = np.random.randint(h)
        x0 = max(x - length // 2, 0)
        x1 = min(x + length // 2, w)
        y0 = max(y - length // 2, 0)
        y1 = min(y + length // 2, h)
        mask[i, y0:y1, x0:x1] = 0
    
    mask = mask[:, np.newaxis, :, :]
    mask = np.broadcast_to(mask, (batch_size, images.shape[1], h, w))
    mask = torch.from_numpy(mask)
    mask.to(device)
    images.to(device)
    masked_images = images * mask

    return masked_images

def mixup(images, labels, alpha):
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    
    # 混合图像
    mixed_images = lam * images + (1 - lam) * shuffled_images

    # 返回增强图像，对应标签及混合比例
    return mixed_images, labels, shuffled_labels, lam