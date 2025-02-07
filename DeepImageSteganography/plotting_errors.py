import matplotlib.pyplot as plt
import numpy as np


# Load error values
ssim_values, psnr_values = [], []

with open("errors.txt", "r") as f:
    for line in f:
        ssim_val, psnr_val = map(float, line.strip().split(","))
        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)

# Generate x-axis
runs = np.arange(1, len(ssim_values) + 1)

# Plot SSIM and PSNR
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(runs, ssim_values, marker='o', color='b', label='SSIM')
plt.xlabel('Run')
plt.ylabel('SSIM')
plt.title('SSIM Across Runs')
plt.ylim(0.85, 1.0)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(runs, psnr_values, marker='s', color='r', label='PSNR (dB)')
plt.xlabel('Run')
plt.ylabel('PSNR (dB)')
plt.title('PSNR Across Runs')
plt.ylim(30, 50)
plt.legend()

plt.tight_layout()
plt.show()
