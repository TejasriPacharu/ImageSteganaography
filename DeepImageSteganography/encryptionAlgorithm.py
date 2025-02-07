from PIL import Image
import numpy as np

def logistic_scramble(image, S):
    w, h = image.shape[:2]
    flat_image = image.flatten()
    index = np.argsort(S[:len(flat_image)])
    scramble_image = flat_image[index]
    scramble_image = scramble_image.reshape((w, h, -1)) if image.ndim == 3 else scramble_image.reshape((w, h))
    return scramble_image

def recursive_encryption(image, width, height, S):
    nh, nw = height // 2, width // 2
    if nh < 1 or nw < 1:
        return image
    recursive_encryption(image[:nw, :nh], nw, nh, S)
    image[:nw, :nh] = logistic_scramble(image[:nw, :nh], S) 
    recursive_encryption(image[nw:width, :nh], nw, nh, S) 
    image[nw:width, :nh] = logistic_scramble(image[nw:width, :nh], S) 
    recursive_encryption(image[:nw, nh:height], nw, nh, S)
    image[:nw, nh:height] = logistic_scramble(image[:nw, nh:height], S) 
    recursive_encryption(image[nw:width, nh:height], nw, nh, S)
    image[nw:width, nh:height] = logistic_scramble(image[nw:width, nh:height], S)    
    image[:width, :height] = logistic_scramble(image[:width, :height], S)
    return image