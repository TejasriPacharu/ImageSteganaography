import cv2
import numpy as np
from PIL import Image
from encryptionAlgorithm import recursive_encryption

# Load Secret Image
secret_image_path = "secret_image.jpeg"
secret_image = cv2.imread(secret_image_path)
secret_image = cv2.cvtColor(secret_image, cv2.COLOR_BGR2RGB)

# Convert Secret Image to NumPy Array
imgarr = np.array(secret_image)

# Generate Chaotic Sequence
chaotic_seq = np.zeros(imgarr.size)
x0, lamb = 0.6, 3.1  # Initial conditions for logistic map
chaotic_seq[0] = x0
for i in range(1, imgarr.size):
    chaotic_seq[i] = lamb * chaotic_seq[i - 1] * (1 - chaotic_seq[i - 1])

# Encrypt Secret Image
encrypted_secret_np = recursive_encryption(imgarr.copy(), imgarr.shape[0], imgarr.shape[1], chaotic_seq)

# Convert Encrypted Image Back to PIL and Save
encrypted_image = Image.fromarray(encrypted_secret_np.astype("uint8"))
encrypted_image.save("encrypted_secret.png")
print("Encrypted secret image saved as 'encrypted_secret.png'")
