from PIL import Image
import numpy as np


def read(path):
    """Reads an image, given its path. Returns the matrix."""
    i = Image.open(path)
    return np.asarray(i, dtype=np.double)


def save(image, path):
    """Save the image in the given path."""
    i = Image.fromarray(image)
    i = i.convert("L")
    i.save(path)
