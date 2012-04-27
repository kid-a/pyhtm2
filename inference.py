## the inference module

import network
import numpy as np
import utils

def inference(uNode, uLayer, uSigma=0, uClass=None):
    if uLayer == network.ENTRY:
        p = EntryInferenceMaker()
        return p.inference(uNode, uSigma)

    elif uLayer == network.INTERMEDIATE:
        p = IntermediateInferenceMaker()
        return p.inference(uNode)
    
    else:
        p = OutputInferenceMaker()
        return p.inference(uNode, uClass)


class InferenceMaker(object):
    def dens_over_coinc(self, uCoincidences, uCurrentInput): pass
    
    def dens_over_groups(self, uY, uPCG):
        return np.dot(uY, uPCG)

    def inference(self, uNode, uSigma=None):
        y = utils.normalize_over_rows(
            self.dens_over_coinc(uNode.coincidences,
                                 uNode.input_msg,
                                 uSigma))

        z = self.dens_over_groups(y, uNode.PCG)
        return z
        

class EntryInferenceMaker(InferenceMaker):
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=1):
        y = np.sum(np.abs(uCoincidences - uCurrentInput)**2,axis=-1)**(1./2)
        #y = np.apply_along_axis(np.linalg.norm, 1, (uCoincidences - uCurrentInput))
        y = np.exp(- np.power(y, 2) / np.power(uSigma, 2))
        return np.array([y])


class IntermediateInferenceMaker(InferenceMaker):
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=None):
        (rows, cols) = uCoincidences.shape
        y = np.array([[]])

        for i in range(rows):
            selected_coincidences = np.array([])

            for j in range(cols):
                selected_coincidences = \
                    np.append(selected_coincidences, 
                              uCurrentInput[j][0][uCoincidences[i,j]])

            y = np.append(y, np.prod(selected_coincidences))
            
        return np.array([y])
    

class OutputInferenceMaker(InferenceMaker):
    def dens_over_classes(self, uY, uPCW):
        return np.dot(uY, uPCW)
    
    def class_post_prob(self, uZ, uClassPriorProb):
        total = np.dot(uZ, uClassPriorProb)
        return uZ * uClassPriorProb / total
    
    def inference(self, uNode):
        y = utils.normalize_over_rows(
            self.dens_over_coinc(uNode.coincidences,
                                 uNode.input_msg,
                                 uSigma))
        
        
        z = self.dens_over_classes(y, uNode.PCW)
        p = self.class_post_prob(z, uNode.cls_prior_prob)
        #uNode.output_msg = p        
        return p


if __name__ == "__main__":
    from network import Node
    from spatial_clustering import EntrySpatialPooler
    
    n = Node()

    ## train entry node
    n = network.Node()
    s = EntrySpatialPooler(0, 5)
    n.input_msg = np.array([1,2,3,4])
    s.train_node(n)
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    s.train_node(n)

    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    n.input_msg = np.array([0,0,0,0])
    s.train_node(n)

    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    n.input_msg = np.array([1,1,1,1])
    s.train_node(n)
    
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    n.input_msg = np.array([1,2,3,4])
    s.train_node(n)
    
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    print "Now making inference"
    e = EntryInferenceMaker()
    e.inference(n, 25.0)

    print n.output_msg
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
