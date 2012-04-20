## the inference module

import layer
import numpy as np


def dens_over_coinc(uCoincidences, uCurrentInput, 
                    uCurrentLayer=layer.INTERMEDIATE, uSigma=1):

    if uCurrentLayer == layer.ENTRY:
        y = np.apply_along_axis(np.linalg.norm, 1, (uCoincidences - uCurrentInput))
        y = np.exp(- np.power(y, 2) / np.power(uSigma, 2))
        
    else:
        (rows, cols) = uCoincidences.shape
        
        y = np.array([])
    
        for i in range(rows):
            selected_coincidences = np.array([])
            
            for j in range(cols):                
            
                selected_coincidences = \
                    np.append(selected_coincidences, 
                              uCurrentInput[j][uCoincidences[i][j]])

            y = np.append(y, np.prod(selected_coincidences))

    return y


def dens_over_groups():
    pass


def dens_over_classes():
    pass
    
