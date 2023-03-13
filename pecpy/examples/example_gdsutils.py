import matplotlib.pyplot as plt
from pecpy.gdsutils import *
from pecpy.core import *

# Convert (greyscale, two-toned) image to gds.
img = misc.imread(get_resource("cheese_mini.png"), flatten=True)
lib = img2gds(img, tolerance=10, min_area=16, levels=2)
with open("example.gds", 'wb') as stream:
    lib.save(stream)

# Convert gds to image.
img2 = gds2img(256, 256, "example.gds", decenter=False)

# Compare visually.
plt.figure(figsize=(5, 3))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Converted")  # The tolerance is set too high to make the converted/original polygons distinguishable
plt.imshow(img2, cmap='gray')
plt.tight_layout()
plt.show()