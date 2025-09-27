import numpy as np
from skimage import io, color, measure
import svgwrite

# Load the image
img = io.imread('depositphotos_324394814-stock-illustration-web-icon-website-vector-icon.jpg')

# If image has alpha channel, convert to RGB
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)

# Convert to grayscale
gray = color.rgb2gray(img)

# Find contours with lower threshold for more details
contours = measure.find_contours(gray, 0.3)

# Create SVG
dwg = svgwrite.Drawing('depositphotos_traced.svg', size=(img.shape[1], img.shape[0]))

for contour in contours:
    # Create path data
    path_data = 'M ' + ' L '.join(f'{int(p[1])},{int(p[0])}' for p in contour) + ' Z'
    dwg.add(dwg.path(d=path_data, fill='none', stroke='black', stroke_width=1))

dwg.save()
print("SVG traced and saved as depositphotos_traced.svg")