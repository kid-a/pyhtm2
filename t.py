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
    
    
    for i in range(10):        
        print i
        image = usps.read("data_sets/train100/0/" + str(i+1) + ".bmp")
        htm.expose(image)
        htm.layers[0].train({'temporal_gap' : False})
    
    htm.layers[0].finalize()

    htm.layers[0].nodes[0][0].input_channel.put("clone_state")
    print htm.layers[0].nodes[0][0].output_channel.get()

    for i in range(2):        
        print i
        image = usps.read("data_sets/train100/0/" + str(i+1) + ".bmp")
        htm.expose(image)
        htm.layers[0].inference()
        htm.layers[0].nodes[0][0].input_channel.put("get_output")
        print htm.layers[0].nodes[0][0].output_channel.get()
    

