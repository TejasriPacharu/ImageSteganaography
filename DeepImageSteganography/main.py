import torch
import torchvision.transforms as transforms
import cv2

# Load Cover Image
cover_image_path = "cover_image.jpeg"
cover_image = cv2.imread(cover_image_path)
cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)

# Load Encrypted Secret Image
encrypted_secret_path = "encrypted_secret.png"
encrypted_secret_image = cv2.imread(encrypted_secret_path)
encrypted_secret_image = cv2.cvtColor(encrypted_secret_image, cv2.COLOR_BGR2RGB)

# Define Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Convert to Tensors
cover_image_tensor = transform(cover_image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
encrypted_secret_tensor = transform(encrypted_secret_image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# Print Tensor Shapes
print("Cover Image Tensor Shape:", cover_image_tensor.shape)
print("Encrypted Secret Image Tensor Shape:", encrypted_secret_tensor.shape)
