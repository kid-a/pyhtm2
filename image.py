from PIL import Image
import numpy as np

import layer

WHITE = 255
BLACK = 0


def read(path):
    """Reads an image, given its path. Returns the matrix."""
    i = Image.open(path)
    return np.asarray(i, dtype=np.double)


def save(image, path):
    """Save the image in the given path."""
    i = Image.fromarray(image)
    i = i.convert("L")
    i.save(path)


def make_training_seq(uImage, uLayer, uWindowSize=(4,4)):
    """Make a training sequence out of an image."""
    sequence = []

    if uLayer == layer.ENTRY:

        ## extract the bounding box
        image = uImage[np.any(uImage - WHITE, 1),:]
        image = image[:,np.any(image - WHITE, 0)]

        ## now, pad the image with white space
        (rows, cols) = image.shape

        image = np.hstack((WHITE * np.ones((rows, uWindowSize[1])),
                           image,
                           WHITE * np.ones((rows, uWindowSize[1]))))

        image = np.vstack((WHITE * np.ones((uWindowSize[0], cols + 2 * uWindowSize[1])),
                           image,
                           WHITE * np.ones((uWindowSize[0], cols + 2 * uWindowSize[1]))))

        ## perform the scans
        (rows, cols) = image.shape        
        vertical_sequence = range(rows - uWindowSize[0] + 1)
        horizontal_sequence = range(cols - uWindowSize[1] + 1)
        
        ## horizontal scan
        horizontal_sequence.reverse()

        for i in vertical_sequence:
            for j in horizontal_sequence:
                sequence.append(image[i:(i + uWindowSize[0]), 
                                      j:(j + uWindowSize[1])])

            horizontal_sequence.reverse()

        ## vertical scan
        horizontal_sequence = range(cols - uWindowSize[1] + 1)
        horizontal_sequence.reverse()
        vertical_sequence.reverse()
        
        for j in horizontal_sequence:
            for i in vertical_sequence:
                sequence.append(image[i:(i + uWindowSize[0]), 
                                      j:(j + uWindowSize[1])])

                vertical_sequence.reverse()
                
    elif uLayer == layer.INTERMEDIATE:
        pass
    
    elif uLayer == layer.OUTPUT:
        pass

    return sequence


if __name__ == "__main__":
    i = read("data_sets/train/2/1.bmp")
    sequence = make_training_seq(i, 0)
    
    # i = 0
    # for s in sequence:
    #     save(s, 'im' + str(i) + '.bmp')
    #     i += 1

    
