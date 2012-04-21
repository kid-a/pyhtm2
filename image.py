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
    temporal_gaps = [0]

    if uLayer == layer.ENTRY:
        ## crop the foreground object
        image = uImage[np.any(uImage - WHITE, 1),:]
        image = image[:,np.any(image - WHITE, 0)]
        (rows, cols) = image.shape

        ## now, pad the image with white space
        image = np.hstack((WHITE * np.ones((rows, uWindowSize[1])),
                           image,
                           WHITE * np.ones((rows, uWindowSize[1]))))

        image = np.vstack((WHITE * np.ones((uWindowSize[0], 
                                            cols + 2 * uWindowSize[1])),
                           image,
                           WHITE * np.ones((uWindowSize[0], 
                                            cols + 2 * uWindowSize[1]))))

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

        ## mark the transition between the two scans
        ## as a temporal gap
        temporal_gaps = [len(sequence)]

        ## vertical scan
        horizontal_sequence = range(cols - uWindowSize[1] + 1)
        horizontal_sequence.reverse()
        vertical_sequence.reverse()
        
        for j in horizontal_sequence:
            for i in vertical_sequence:
                sequence.append(image[i:(i + uWindowSize[0]), 
                                      j:(j + uWindowSize[1])])

            vertical_sequence.reverse()

        return (sequence, temporal_gaps)

    else:
        ## crop the foreground object
        (rows, cols) = uImage.shape
        image = uImage[np.any(uImage - WHITE, 1),:]
        image = image[:,np.any(image - WHITE, 0)]
        (foreground_rows, foreground_cols) = image.shape
        
        ## now, put the foreground object in the lower-left corner
        image = np.hstack((WHITE * np.ones((rows, cols - foreground_cols)),
                           image))
        
        image = np.vstack((WHITE * np.ones((rows - foreground_rows, cols)),
                           image))

        ## !FIXME implement the scan
        
        if uLayer == layer.OUTPUT:
            return (sequence, temporal_gaps)
                
        elif uLayer == layer.INTERMEDIATE:
            return (sequence, temporal_gaps)


if __name__ == "__main__":
    ## some tests
    i = read("data_sets/train/1/1.bmp")
    (sequence, temporal_gaps) = make_training_seq(i, layer.ENTRY)
    
    print temporal_gaps
    
    i = 0
    for s in sequence:
        save(s, 'im' + str(i) + '.bmp')
        i += 1

    ##(sequence, temporal_gaps) = make_training_seq(i, layer.INTERMEDIATE)

    
