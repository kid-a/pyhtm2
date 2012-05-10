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


## location of usps data sets --------------------------------------------------
USPS_TEST_SET = "data_sets/test"
USPS_TRAIN_SET = "data_sets/train"
USPS_TRAIN100_SET = "data_sets/train100"
USPS_TRAIN1000_SET = "data_sets/train1000"


if __name__ == "__main__":
    ## print arrays in full
    np.set_printoptions(threshold='nan')

    print "*** Select the training set: ***"
    print "1. Train HTM on USPS train100 training set"
    print "2. Train HTM on USPS train1000 training set"
    print "3. Train HTM on USPS full training set (over 7000 elements)"
    print "4. Load HTM from file"
    print "5. Quit"
    choice = int(raw_input())
    
    if choice == 1 or choice == 2 or choice == 3:
        builder = NetworkBuilder(config.usps_net)
        htm = builder.build()
    
        htm.start()
        t0 = time.time()
    
        print
        print "*** Training HTM **"
        seq_count = {}
        
        if choice == 1: directory = "train100"
        elif choice == 2: directory = "train1000"
        else: directory = "train"

        sequences = usps.get_training_sequences(directory, uSeqCount=seq_count)
        
        print "Starting training..."
        import profile
        
        profile.runctx('htm.train(sequences)', globals(), {'htm':htm,
                                                           'sequences':sequences})
        
        print "Saving network on file..."
        try: os.mkdir("usps/" + directory)
        except: pass

        save(htm, "usps/" + directory + "/")

        print "*** Summary **"
        print "Number of training sequences generated:"
        print " * Entry layer:        ", seq_count[network.ENTRY]
        print " * Intermediate layer: ", seq_count[network.INTERMEDIATE]
        print " * Output layer:       ", seq_count[network.OUTPUT]
        print "Training completed in ", time.time() - t0, "seconds"
        
    elif choice == 4:
        print "Enter the directory:"
        directory = raw_input()
        htm = load(directory)
        
    else:
        exit(0)

    print "*** HTM Testing ***"
    print "1. Test HTM on single input"
    print "2. Test HTM on USPS train100 training set"
    print "3. Test HTM on USPS train1000 training set"
    print "4. Test HTM on USPS full test set (over 2000 elements)"
    print "5. Quit"

    choice = int(raw_input())

    print    
    print "*** Testing HTM ***"
    if choice == 1:
        print htm.inference(read("data_sets/test/0/1.bmp"))
        
    elif choice == 2 or choice == 3 or choice == 4:
        if choice == 2: directory = USPS_TRAIN100_SET
        elif choice == 3: directory = USPS_TRAIN1000_SET
        elif choice == 4: directory = USPS_TEST_SET
        
        classes = os.listdir(directory)
        
        total = 0
        correct = 0
        for c in classes:
            current_class = int(c)

            for i in os.listdir(directory + '/' + c):
                total += 1

                res = np.array(htm.inference(read(directory + '/' + c + '/' + i)))
                res = np.argmax(res)

                if res == current_class:
                    correct += 1
                
        print "Total:", total
        print "Correct:", correct
        print "Correctness ratio:", correct/float(total)

    else:
        exit(0)


