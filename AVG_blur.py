import numpy as np

def AVG_filter(face, kernel_size=25):
    h, w, c = face.shape
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size**2)
    kh, kw = kernel.shape
    result = np.zeros_like(face, dtype=float)

    pad_h = kh // 2
    pad_w = kw // 2

    # zero padding
    padded = np.pad(face, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+kh, x:x+kw, ch]
                result[y, x, ch] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)