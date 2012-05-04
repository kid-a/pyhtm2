## global import ---------------------------------------------------------------
import numpy as np
import time
import os


## local import ----------------------------------------------------------------
from network import NetworkBuilder as NetworkBuilder
from network import save
from network import load
from usps import read as read
# from usps import save as save
import network
import usps
import config
import debug


TRAINING_SET = "train100"
TEST_SET = "data_sets/test"

if __name__ == "__main__":
    builder = NetworkBuilder(config.test_net)
    htm = builder.build()
    htm.start()
    
    ## train layer 0
    for j in range(5):
        for i in range(10):
            image = usps.read("data_sets/train100/" + str(j) + "/" + str(i+1) + ".bmp")
            htm.expose(image)
            if i == 0: htm.layers[0].train({'temporal_gap' : True})
            else: htm.layers[0].train({'temporal_gap' : False})
    
    htm.layers[0].finalize()

    ## train layer 1
    for j in range(5):
        for i in range(10):
            image = usps.read("data_sets/train100/" + str(j) + "/" + str(i+1) + ".bmp")
            htm.expose(image)
            htm.layers[0].inference()
            htm.propagate(0, 1)
            
            if i == 0: htm.layers[1].train({'temporal_gap' : True})
            else: htm.layers[1].train({'temporal_gap' : False})

    htm.layers[1].finalize()

    ## train layer 3
    for j in range(3):
        for i in range(5):
            print i
            image = usps.read("data_sets/train100/" + str(j) + "/" + str(i+1) + ".bmp")
            htm.expose(image)
            htm.layers[0].inference()
            htm.propagate(0, 1)
            htm.layers[1].inference()
            htm.propagate(1, 2)
            
            htm.layers[2].train({'class' : j})

    htm.layers[2].finalize()

    for i in range(1):
        for j in range(1):
            htm.layers[2].nodes[i][j].input_channel.put("clone_state")
            print i, j, htm.layers[2].nodes[i][j].output_channel.get()
            

    # for i in range(1):        
    #     print i
    #     image = usps.read("data_sets/train100/0/" + str(i+1) + ".bmp")
    #     htm.expose(image)
    #     htm.layers[0].inference()
    #     # htm.layers[0].nodes[0][0].input_channel.put("get_output")
    #     # print "out, first layer: ", htm.layers[0].nodes[0][0].output_channel.get()
    #     htm.propagate(0, 1)
    #     htm.layers[1].inference()
    #     htm.layers[1].nodes[0][0].input_channel.put("get_output")
    #     print "out, second layer: ", htm.layers[1].nodes[0][0].output_channel.get()
