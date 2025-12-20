import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy import signal
from scipy.linalg import toeplitz
import os

# Create output directory
os.makedirs("img", exist_ok=True)

print("Generating Lab09 images...")

# ============================================================================
# PART 1: 1D FFT Example (from notebook 1)
# ============================================================================

# Create signal: sum of two sinusoids
dt = 1e-2  # [s]
t = np.arange(0, 5, dt)
f = np.sin(5 * 2 * np.pi * t) + 0.5 * np.sin(13.5 * 2 * np.pi * t)

# Plot time domain signal
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(t, f, linewidth=2)
ax.set_xlabel("t [s]", fontsize=12)
ax.set_ylabel("f(t)", fontsize=12)
ax.set_title("Time Domain Signal: f(t) = sin(5·2πt) + 0.5·sin(13.5·2πt)", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/lab09_signal_time_domain.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_signal_time_domain.png")

# Compute FFT
F = np.fft.fft(f)
freq = np.fft.fftfreq(len(t), d=dt)

# Reorder frequencies
F_shift = np.fft.fftshift(F)
freq_shift = np.fft.fftshift(freq)

# Plot frequency domain
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(freq_shift, np.absolute(F_shift), linewidth=2)
ax.set_xlabel("Frequency [Hz]", fontsize=12)
ax.set_ylabel("Magnitude", fontsize=12)
ax.set_title("Frequency Domain (FFT): Clear Peaks at 5 Hz and 13.5 Hz", fontsize=14)
ax.grid(True, alpha=0.3)
ax.axvline(x=5, color='r', linestyle='--', alpha=0.7, label='5 Hz')
ax.axvline(x=13.5, color='g', linestyle='--', alpha=0.7, label='13.5 Hz')
ax.axvline(x=-5, color='r', linestyle='--', alpha=0.7)
ax.axvline(x=-13.5, color='g', linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.savefig("img/lab09_signal_frequency_domain.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_signal_frequency_domain.png")

# Reconstruct signal with IFFT
f_reconstructed = np.fft.ifft(F)
f_reconstructed_real = np.real(f_reconstructed)

# Plot original vs reconstructed
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(t, f, label="Original f(t)", linewidth=2)
ax.plot(t, f_reconstructed_real, "--", label="Reconstructed (IFFT)", linewidth=2, alpha=0.8)
ax.set_xlabel("t [s]", fontsize=12)
ax.set_ylabel("f(t)", fontsize=12)
ax.set_title("Signal Reconstruction: Original vs IFFT(FFT(f))", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/lab09_signal_reconstruction.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_signal_reconstruction.png")

# ============================================================================
# PART 2: 2D FFT Example (from notebook 1)
# ============================================================================

# Create 2D signal with horizontal stripes
x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
xx, yy = np.meshgrid(x, y)
z = np.sin(2*np.pi*1*xx)

# FFT
Z = np.fft.fft2(z)
Z_shift = np.fft.fftshift(Z)

freq_x_shift = np.fft.fftshift(np.fft.fftfreq(len(x), d=0.1))
freq_y_shift = np.fft.fftshift(np.fft.fftfreq(len(y), d=0.1))
limits = [freq_x_shift[0], freq_x_shift[-1], freq_y_shift[0], freq_y_shift[-1]]

# iFFT
z_reconstructed = np.fft.ifft2(Z)

# Plot 2D FFT
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
im0 = axs[0].imshow(z, cmap='viridis')
axs[0].set_title("Original Signal (Horizontal Stripes)", fontsize=12)
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.log10(np.absolute(Z_shift) + 1), extent=limits, cmap='hot')
axs[1].set_title("FFT Magnitude (Log Scale)", fontsize=12)
axs[1].set_xlabel("Frequency X", fontsize=10)
axs[1].set_ylabel("Frequency Y", fontsize=10)
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(np.real(z_reconstructed), cmap='viridis')
axs[2].set_title("Reconstructed Signal (IFFT)", fontsize=12)
axs[2].axis('off')
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.savefig("img/lab09_2d_fft_horizontal.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_2d_fft_horizontal.png")

# Radial frequency example
z_radial = np.sin(10 * np.sqrt(xx**2 + yy**2))
Z_radial = np.fft.fft2(z_radial)
Z_radial_shift = np.fft.fftshift(Z_radial)
z_radial_reconstructed = np.fft.ifft2(Z_radial)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
im0 = axs[0].imshow(z_radial, cmap='viridis')
axs[0].set_title("Radial Signal (Concentric Waves)", fontsize=12)
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.log10(np.absolute(Z_radial_shift) + 1), extent=limits, cmap='hot')
axs[1].set_title("FFT Magnitude (Circular Ring)", fontsize=12)
axs[1].set_xlabel("Frequency X", fontsize=10)
axs[1].set_ylabel("Frequency Y", fontsize=10)
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(np.real(z_radial_reconstructed), cmap='viridis')
axs[2].set_title("Reconstructed Signal", fontsize=12)
axs[2].axis('off')
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.savefig("img/lab09_2d_fft_radial.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_2d_fft_radial.png")

# ============================================================================
# PART 3: 1D Convolution Examples (from notebook 2)
# ============================================================================

# Create square wave signal
v = np.zeros(100)
v[50:75] = 1

# Define kernels
k_boxcar = np.ones(10) / 10
k_gaussian = signal.windows.gaussian(20, std=3)
k_gaussian = k_gaussian / np.sum(k_gaussian)
k_derivative = np.array([-1, 1])

# Compute convolutions using Toeplitz matrix
def convolve_toeplitz(signal_vec, kernel):
    n_signal = len(signal_vec)
    n_kernel = len(kernel)
    output_len = n_signal + n_kernel - 1
    
    # Pad kernel
    k_padded = np.zeros(output_len)
    k_padded[:n_kernel] = kernel
    
    # Create Toeplitz matrix
    first_row = np.zeros(n_signal)
    first_row[0] = k_padded[0]
    K = toeplitz(k_padded, first_row)
    
    # Convolve
    return K @ signal_vec

v_conv_boxcar = convolve_toeplitz(v, k_boxcar)
v_conv_gaussian = convolve_toeplitz(v, k_gaussian)
v_conv_derivative = convolve_toeplitz(v, k_derivative)

# Plot convolution results
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Original signal
axs[0, 0].plot(v, 'b-', linewidth=2)
axs[0, 0].set_title("Original Square Wave Signal", fontsize=12)
axs[0, 0].set_ylabel("Amplitude", fontsize=10)
axs[0, 0].grid(True, alpha=0.3)

# Boxcar (moving average)
axs[0, 1].plot(v_conv_boxcar, 'g-', linewidth=2)
axs[0, 1].set_title("Boxcar Kernel (Moving Average)", fontsize=12)
axs[0, 1].set_ylabel("Amplitude", fontsize=10)
axs[0, 1].grid(True, alpha=0.3)

# Gaussian
axs[1, 0].plot(v_conv_gaussian, 'r-', linewidth=2)
axs[1, 0].set_title("Gaussian Kernel (Smooth Blur)", fontsize=12)
axs[1, 0].set_xlabel("Sample Index", fontsize=10)
axs[1, 0].set_ylabel("Amplitude", fontsize=10)
axs[1, 0].grid(True, alpha=0.3)

# Derivative
axs[1, 1].plot(v_conv_derivative, 'm-', linewidth=2)
axs[1, 1].set_title("Derivative Kernel [-1, 1]", fontsize=12)
axs[1, 1].set_xlabel("Sample Index", fontsize=10)
axs[1, 1].set_ylabel("Amplitude", fontsize=10)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("img/lab09_1d_convolution_kernels.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_1d_convolution_kernels.png")

# FFT-based convolution comparison
V_fft = np.fft.fft(v)
K_fft = np.fft.fft(k_boxcar, n=len(v))
VK_fft = V_fft * K_fft
v_conv_fft = np.real(np.fft.ifft(VK_fft))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Signal spectrum
axs[0, 0].plot(np.fft.fftshift(np.abs(V_fft)), 'b-', linewidth=2)
axs[0, 0].set_title("Signal FFT Magnitude", fontsize=12)
axs[0, 0].set_ylabel("Magnitude", fontsize=10)
axs[0, 0].grid(True, alpha=0.3)

# Kernel spectrum
axs[0, 1].plot(np.fft.fftshift(np.abs(K_fft)), 'g-', linewidth=2)
axs[0, 1].set_title("Kernel FFT Magnitude (Low-Pass)", fontsize=12)
axs[0, 1].set_ylabel("Magnitude", fontsize=10)
axs[0, 1].grid(True, alpha=0.3)

# Product spectrum
axs[1, 0].plot(np.fft.fftshift(np.abs(VK_fft)), 'r-', linewidth=2)
axs[1, 0].set_title("Product FFT Magnitude", fontsize=12)
axs[1, 0].set_xlabel("Frequency Index", fontsize=10)
axs[1, 0].set_ylabel("Magnitude", fontsize=10)
axs[1, 0].grid(True, alpha=0.3)

# Reconstructed signal
axs[1, 1].plot(v, 'b--', alpha=0.5, label='Original', linewidth=2)
axs[1, 1].plot(v_conv_fft, 'r-', label='FFT Convolution', linewidth=2)
axs[1, 1].set_title("Convolution via FFT", fontsize=12)
axs[1, 1].set_xlabel("Sample Index", fontsize=10)
axs[1, 1].set_ylabel("Amplitude", fontsize=10)
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("img/lab09_fft_convolution_spectrum.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_fft_convolution_spectrum.png")

# ============================================================================
# PART 4: 2D Convolution Examples (from notebook 3)
# ============================================================================

# Load image (or create synthetic if not available)
try:
    image_path = "note/Lab09/NYlibrary.png"
    v_img = np.mean(imread(image_path), axis=2)
except:
    # Create synthetic image if file not found
    print("  (Creating synthetic image as NYlibrary.png not found)")
    x_img = np.linspace(-5, 5, 300)
    y_img = np.linspace(-5, 5, 300)
    xx_img, yy_img = np.meshgrid(x_img, y_img)
    v_img = np.sin(xx_img) * np.cos(yy_img) + 0.5 * np.sin(2*xx_img + yy_img)
    v_img = (v_img - v_img.min()) / (v_img.max() - v_img.min()) * 255

vmin = v_img.min()
vmax = v_img.max()

# Define 2D kernels
kernel_blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
kernel_blur = kernel_blur / np.sum(kernel_blur)

kernel_edge_laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

kernel_sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
kernel_sobel_vertical = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# Apply convolutions using scipy
v_blur = signal.convolve(v_img, kernel_blur, mode='same')
v_edge = signal.convolve(v_img, kernel_edge_laplacian, mode='same')
v_sharpen = signal.convolve(v_img, kernel_sharpen, mode='same')
v_sobel_h = signal.convolve(v_img, kernel_sobel_horizontal, mode='same')
v_sobel_v = signal.convolve(v_img, kernel_sobel_vertical, mode='same')

# Plot 2D convolution results
fig, axs = plt.subplots(2, 3, figsize=(16, 10))

axs[0, 0].imshow(v_img, cmap='gray', vmin=vmin, vmax=vmax)
axs[0, 0].set_title("Original Image", fontsize=12)
axs[0, 0].axis('off')

axs[0, 1].imshow(v_blur, cmap='gray', vmin=vmin, vmax=vmax)
axs[0, 1].set_title("Blur Filter (Averaging)", fontsize=12)
axs[0, 1].axis('off')

axs[0, 2].imshow(v_sharpen, cmap='gray')
axs[0, 2].set_title("Sharpen Filter", fontsize=12)
axs[0, 2].axis('off')

axs[1, 0].imshow(v_edge, cmap='gray')
axs[1, 0].set_title("Laplacian Edge Detection", fontsize=12)
axs[1, 0].axis('off')

axs[1, 1].imshow(v_sobel_h, cmap='gray')
axs[1, 1].set_title("Sobel Horizontal Edges", fontsize=12)
axs[1, 1].axis('off')

axs[1, 2].imshow(v_sobel_v, cmap='gray')
axs[1, 2].set_title("Sobel Vertical Edges", fontsize=12)
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig("img/lab09_2d_convolution_filters.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_2d_convolution_filters.png")

# FFT analysis of kernels
fig, axs = plt.subplots(2, 3, figsize=(16, 10))

# Blur kernel spectrum
k_blur_fft = np.fft.fft2(kernel_blur, s=v_img.shape)
k_blur_fft_shift = np.fft.fftshift(k_blur_fft)
axs[0, 0].imshow(kernel_blur, cmap='gray')
axs[0, 0].set_title("Blur Kernel", fontsize=12)
axs[0, 0].axis('off')

axs[1, 0].imshow(np.log10(np.abs(k_blur_fft_shift) + 1e-10), cmap='hot')
axs[1, 0].set_title("Blur FFT (Low-Pass)", fontsize=12)
axs[1, 0].axis('off')

# Edge kernel spectrum
k_edge_fft = np.fft.fft2(kernel_edge_laplacian, s=v_img.shape)
k_edge_fft_shift = np.fft.fftshift(k_edge_fft)
axs[0, 1].imshow(kernel_edge_laplacian, cmap='gray')
axs[0, 1].set_title("Laplacian Kernel", fontsize=12)
axs[0, 1].axis('off')

axs[1, 1].imshow(np.log10(np.abs(k_edge_fft_shift) + 1e-10), cmap='hot')
axs[1, 1].set_title("Laplacian FFT (High-Pass)", fontsize=12)
axs[1, 1].axis('off')

# Sharpen kernel spectrum
k_sharpen_fft = np.fft.fft2(kernel_sharpen, s=v_img.shape)
k_sharpen_fft_shift = np.fft.fftshift(k_sharpen_fft)
axs[0, 2].imshow(kernel_sharpen, cmap='gray')
axs[0, 2].set_title("Sharpen Kernel", fontsize=12)
axs[0, 2].axis('off')

axs[1, 2].imshow(np.log10(np.abs(k_sharpen_fft_shift) + 1e-10), cmap='hot')
axs[1, 2].set_title("Sharpen FFT Spectrum", fontsize=12)
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig("img/lab09_kernel_frequency_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab09_kernel_frequency_analysis.png")

print("\n" + "="*60)
print("All Lab09 images generated successfully!")
print("="*60)
print("\nGenerated images:")
print("  1. lab09_signal_time_domain.png - Time domain signal")
print("  2. lab09_signal_frequency_domain.png - FFT spectrum with peaks")
print("  3. lab09_signal_reconstruction.png - Original vs IFFT reconstruction")
print("  4. lab09_2d_fft_horizontal.png - 2D FFT with horizontal stripes")
print("  5. lab09_2d_fft_radial.png - 2D FFT with radial pattern")
print("  6. lab09_1d_convolution_kernels.png - Different 1D kernels")
print("  7. lab09_fft_convolution_spectrum.png - FFT-based convolution")
print("  8. lab09_2d_convolution_filters.png - 2D image filters")
print("  9. lab09_kernel_frequency_analysis.png - Kernel frequency spectra")
