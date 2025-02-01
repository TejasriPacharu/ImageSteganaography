import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from hiddenNetwork import SwinSteganography

hiding_network = SwinSteganography()

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

stego_image_tensor = hiding_network(cover_image_tensor, encrypted_secret_tensor)

stego_image = stego_image_tensor.squeeze(0).detach().permute(1, 2, 0).numpy()  # Convert (1,3,H,W) -> (H,W,3)
stego_image = (stego_image * 255).astype("uint8")  # Rescale to 0-255

# Save the Stego Image
stego_pil = Image.fromarray(stego_image)
stego_pil.save("stego_image.png")

print("✅ Stego image successfully created and saved as 'stego_image.png'")