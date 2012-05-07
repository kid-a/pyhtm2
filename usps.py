## global import ---------------------------------------------------------------
from multiprocessing import Pool
from PIL import Image
import numpy as np
import os

## local import ----------------------------------------------------------------
import network

BASEPATH = "data_sets"

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
    

def make_training_seq(uImage, uLayer, uWindowSize=(4,4), uClass=None):
    """Make a training sequence out of an image."""
    sequence = []

    if uLayer == network.ENTRY:
        ## crop the foreground object
        (orig_rows, orig_cols) = uImage.shape
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
                
                if i == 0 and j == 0: temporal_gap = True
                else: temporal_gap = False

                img = image[i:(i + uWindowSize[0]), 
                            j:(j + uWindowSize[1])]
                
                img = np.hstack((img, 
                                 WHITE * np.ones((uWindowSize[0], 
                                                  orig_cols - uWindowSize[1]))))
                
                img = np.vstack((img,
                                 WHITE * np.ones((orig_rows - uWindowSize[0],
                                                  orig_cols))))
                    
                sequence.append((img, {'temporal_gap' : temporal_gap}))

            horizontal_sequence.reverse()

        ## vertical scan
        horizontal_sequence = range(cols - uWindowSize[1] + 1)
        horizontal_sequence.reverse()
        vertical_sequence.reverse()
        
        for j in horizontal_sequence:
            for i in vertical_sequence:

                if i == 0 and j == 0: temporal_gap = True
                else: temporal_gap = False

                img = image[i:(i + uWindowSize[0]), 
                            j:(j + uWindowSize[1])]

                img = np.hstack((img, 
                                 WHITE * np.ones((uWindowSize[0], 
                                                  orig_cols - uWindowSize[1]))))
                
                img = np.vstack((img,
                                 WHITE * np.ones((orig_rows - uWindowSize[0],
                                                  orig_cols))))

                sequence.append((img, {'temporal_gap' : temporal_gap}))

            vertical_sequence.reverse()

        return sequence

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
        first = True

        while True:
            while True:
                if uLayer == network.INTERMEDIATE:
                    if first: 
                        temporal_gap = True
                        first = False

                    else: temporal_gap = False
                    sequence.append((image, {'temporal_gap' : temporal_gap}))
                                        
                else: ## uLayer == network.OUTPUT
                    sequence.append((image, {'class' : uClass}))

                ## move the object
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
            return sequence

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
        first = True

        while True:
            while True:
                if first: 
                    temporal_gap = True
                    first = False

                else: temporal_gap = False
                sequence.append((image, {'temporal_gap' : temporal_gap}))

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

        return sequence


def make_sequence(uClasses, uPath, uType, uSeqPerClass=0):
    sequence = []

    for c in uClasses:
        paths = os.listdir(uPath + '/' + c)
        paths.sort()
        
        if uSeqPerClass != 0:
            paths = paths[0:uSeqPerClass]
        
        for i in paths:
            image = read(uPath + '/' + c + '/' + i)
            sequence.extend(make_training_seq(image, uType, uClass=int(c)))
        
    return sequence


def seq_generator(uClasses, uPath, uType, uSeqCount, uSeqPerClass=0):
    for c in uClasses:
        paths = os.listdir(uPath + '/' + c)
        paths.sort()
        paths.reverse()
        
        if uSeqPerClass != 0:
            paths = paths[0:uSeqPerClass]
                    
        for i in paths:
            image = read(uPath + '/' + c + '/' + i)
            sequence = make_training_seq(image, uType, uClass=int(c))
            for s in sequence:
                uSeqCount[uType] += 1
                yield s


def get_training_sequences(uDir, uSeqPerClass=0, uSeqCount={}, make_generators=True):
    path = BASEPATH + '/' + uDir

    sequences = {network.ENTRY : [],
                 network.INTERMEDIATE : [],
                 network.OUTPUT : []}

    numbers = os.listdir(path)
    numbers.sort()
    numbers.reverse()

    uSeqCount[network.ENTRY] = 0
    uSeqCount[network.INTERMEDIATE] = 0
    uSeqCount[network.OUTPUT] = 0

    if make_generators:
        sequences[network.ENTRY] = seq_generator(numbers, path, 
                                                 network.ENTRY, 
                                                 uSeqCount, 
                                                 uSeqPerClass)
        
        sequences[network.INTERMEDIATE] = seq_generator(numbers, path, 
                                                        network.INTERMEDIATE, 
                                                        uSeqCount, 
                                                        uSeqPerClass)

        sequences[network.OUTPUT] = seq_generator(numbers, path, 
                                                  network.OUTPUT,
                                                  uSeqCount, 
                                                  uSeqPerClass)
        return sequences
        
    else:

        pool = Pool(3)
        result_entry = pool.apply_async(make_sequence,
                                        [numbers, path, 
                                         network.ENTRY, uSeqPerClass])

        result_intermediate = pool.apply_async(make_sequence,
                                               [numbers, path, 
                                                network.INTERMEDIATE, uSeqPerClass])
        
        result_output = pool.apply_async(make_sequence,
                                         [numbers, path, 
                                          network.OUTPUT, uSeqPerClass])
    
        while not result_entry and \
                not result_intermediate and \
                not result_output:
            pass
    
        sequences[network.ENTRY] = result_entry.get()
        sequences[network.INTERMEDIATE] = result_intermediate.get()
        sequences[network.OUTPUT] = result_output.get()
            
        return sequences


if __name__ == "__main__":
    pass
    #i = read("data_sets/train/5/41.bmp")
    #print len(make_training_seq(i, network.ENTRY))
    # print make_training_seq(i, network.INTERMEDIATE)
    # print make_training_seq(i, network.OUTPUT)

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



    # import os
    # import time
    
    # t0 = time.time()

    # sequences = []
    # basepath = 'data_sets/train/'
    # numbers = os.listdir(basepath)
    # for n in numbers:
    #     for e in os.listdir(basepath + n):
    #         i = read(basepath + n + '/' + e)
    #         (s,t) = make_training_seq(i, network.INTERMEDIATE)
    #         sequences.extend(s)

    # print time.time() - t0, "seconds"
    # print len(sequences)
    
