## global import ---------------------------------------------------------------

## local import ----------------------------------------------------------------
from network import NetworkBuilder as NetworkBuilder
from usps import read as read
# from usps import save as save
import network
import usps
import config
import debug
import time

TRAINING_SET = "train100"

if __name__ == "__main__":
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
    
    print "Testing inference..."
    print htm.inference(read("data_sets/test/0/1.bmp"))

    print "Completed in ", time.time() - t0, "seconds"
