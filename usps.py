from PIL import Image
import numpy as np

import network

WHITE = 255
BLACK = 0

LEFT = 0
RIGHT = 1
UP = 0
DOWN = 1


def read(path):
    """Reads an image, given its path. Returns the matrix."""
    i = Image.open(path)
    return np.asarray(i, dtype=np.double)


def save(image, path):
    """Save the image in the given path."""
    i = Image.fromarray(image)
    i = i.convert("L")
    i.save(path)


def crop(image):
    """Extracts the foreground objects out of an image."""
    image = image[np.any(image - WHITE, 1),:]
    image = image[:,np.any(image - WHITE, 0)]
    return image
    

def make_training_seq(uImage, uLayer, uWindowSize=(4,4)):
    """Make a training sequence out of an image."""
    sequence = []
    temporal_gaps = [0]

    if uLayer == network.ENTRY:
        ## crop the foreground object
        image = crop(uImage)
        (rows, cols) = image.shape

        ## now, pad the image with white space
        image = np.hstack((WHITE * np.ones((rows, uWindowSize[1] - 1)),
                           image,
                           WHITE * np.ones((rows, uWindowSize[1] - 1))))

        image = np.vstack((WHITE * np.ones((uWindowSize[0] - 1, 
                                            cols + 2 * uWindowSize[1] - 2)),
                           image,
                           WHITE * np.ones((uWindowSize[0] - 1, 
                                            cols + 2 * uWindowSize[1] - 2))))

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
        temporal_gaps.append(len(sequence))

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

    else: ## uLayer == network.INTERMEDIATE | network.OUTPUT
        ## crop the foreground object
        (rows, cols) = uImage.shape
        image = crop(uImage)
        (foreground_rows, foreground_cols) = image.shape
                
        ## now, put the foreground object in the lower-left corner
        image = np.hstack((WHITE * np.ones((foreground_rows, cols - foreground_cols)),
                           image))
        
        image = np.vstack((WHITE * np.ones((rows - foreground_rows, cols)),
                           image))
                
        ## horizontal scan
        direction = LEFT

        while True:
            while True:
                sequence.append(image)

                if direction == LEFT:
                    ## inner while termination condition
                    if abs(np.sum(image[:,0] - WHITE)) > 0: break
                    
                    ## move the foreground object left
                    image = np.hstack((image[:,1:], WHITE * np.ones((rows, 1))))

                else: ## direction is RIGHT
                    ## inner while termination condition
                    if abs(np.sum(image[:,-1] - WHITE)) > 0: break

                    ## move the foreground object right
                    image = np.hstack((WHITE * np.ones((rows, 1)), image[:,:-1]))
                
            ## outer loop termination condition
            if abs(np.sum(image[0,:] - WHITE)) > 0: break
            
            ## move the foreground object up
            image = np.vstack((image[1:,:], WHITE * np.ones((1, cols))))

            ## invert the direction
            direction = not direction
                        
        if uLayer == network.OUTPUT:
            return (sequence, temporal_gaps)

        ## mark the transition between the two scans as a temporal gap
        temporal_gaps.append(len(sequence))

        ## crop the image, again
        image = crop(uImage)
        (foreground_rows, foreground_cols) = image.shape
                
        ## now, put the foreground object in the lower-left corner, again
        image = np.hstack((WHITE * np.ones((foreground_rows, cols - foreground_cols)),
                           image))
        
        image = np.vstack((WHITE * np.ones((rows - foreground_rows, cols)),
                           image))

        ## vertical scan
        direction = UP

        while True:
            while True:
                sequence.append(image)

                if direction == UP:
                    ## inner while termination condition
                    if abs(np.sum(image[0,:] - WHITE)) > 0: break
                    
                    ## move the foreground object left
                    image = np.vstack((image[1:,:], WHITE * np.ones((1, cols))))

                else: ## direction is DOWN
                    ## inner while termination condition
                    if abs(np.sum(image[-1,:] - WHITE)) > 0: break

                    ## move the foreground object right
                    image = np.vstack((WHITE * np.ones((1, cols)), image[:-1,:]))
                
            ## outer loop termination condition
            if abs(np.sum(image[:,0] - WHITE)) > 0: break
            
            ## move the foreground object LEFT
            image = np.hstack((image[:,1:], WHITE * np.ones((rows, 1))))

            ## invert the direction
            direction = not direction

        return (sequence, temporal_gaps)


if __name__ == "__main__":
    # ## some tests
    # #i = read("data_sets/train/5/41.bmp")
    # i = WHITE * np.ones((24,24))
    # i[0,0] = BLACK
    
    # # (sequence, temporal_gaps) = make_training_seq(i, network.ENTRY)
    # (sequence, temporal_gaps) = make_training_seq(i, network.INTERMEDIATE)
    
    # print temporal_gaps
    
    # i = 0
    # for s in sequence:
    #     save(s, 'im' + str(i) + '.bmp')
    #     i += 1



    import os
    import time
    
    t0 = time.time()

    sequences = []
    basepath = 'data_sets/train/'
    numbers = os.listdir(basepath)
    for n in numbers:
        for e in os.listdir(basepath + n):
            i = read(basepath + n + '/' + e)
            (s,t) = make_training_seq(i, network.INTERMEDIATE)
            sequences.extend(s)

    print time.time() - t0, "seconds"
    print len(sequences)
