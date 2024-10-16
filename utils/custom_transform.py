import torchvision.transforms as transforms
from PIL import Image


# Custom transform to resize image to the nearest size divisible by n
class ResizeToNearestDivisible:
    def __init__(self, n=32):
        self.n = n

    def __call__(self, img):
        width, height = img.size  # Get the original width and height of the image
        new_width = (width // self.n) * self.n  # Find nearest width divisible by n
        new_height = (height // self.n) * self.n  # Find nearest height divisible by n
        return img.resize((new_width, new_height), Image.BILINEAR)  # Resize using bilinear interpolation
