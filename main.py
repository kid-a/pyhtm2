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
    print "*** HTM Training ***"
    print "1. Train HTM on Train100 training set"
    print "2. Load HTM from file"
    choice = int(raw_input())
    
    if choice == 1:
        builder = NetworkBuilder(config.usps_net)
        htm = builder.build()
    
        htm.start()
        t0 = time.time()
    
        print "*** Training htm on", TRAINING_SET, "***"
        sequences = usps.get_training_sequences("train100")
        print "Number of training sequences generated:"
        print " * Entry layer:        ", len(sequences[network.ENTRY])
        print " * Intermediate layer: ", len(sequences[network.INTERMEDIATE])
        print " * Output layer:       ", len(sequences[network.OUTPUT])
        
        print "Starting training..."
        htm.train(sequences)
        
        print "Saving network on file..."
        save(htm, "usps/")
        
    elif choice == 2:
        t0 = time.time()
        htm = load("usps/") 
        
    else:
        print "Abort"
        exit(0)

    print "Training completed in ", time.time() - t0, "seconds"

    print "*** HTM Testing ***"
    print "1. Test HTM on single input"
    print "2. Load HTM on entire test set"

    choice = int(raw_input())
    
    if choice == 1:
        print
        print "*** Testing inference... ***"
        print htm.inference(read("data_sets/test/0/1.bmp"))
        
    elif choice == 2:
        classes = os.listdir(TEST_SET)
        
        total = 0
        correct = 0
        for c in classes:
            current_class = int(c)

            for i in os.listdir(TEST_SET + '/' + c):
                total += 1

                res = np.array(htm.inference(read(TEST_SET + '/' + c + '/' + i)))
                res = np.argmax(res)

                if res == current_class:
                    correct += 1
                
        print "Total:", total
        print "Correct:", correct
        print "Correctness ratio:", correct/float(total)

    else:
        print "Abort"
        exit(0)

    
    
    

