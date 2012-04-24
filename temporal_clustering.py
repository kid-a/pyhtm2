import numpy as np
import utils
import network

class TemporalPooler(object):
    def __init__(self, uMaxGroupSize, uMinGroupSize, uTopNeighbours):
        self.max_group_size = uMaxGroupSize
        self.min_group_size = uMinGroupSize
        self.top_neighbours = uTopNeighbours

    def select_highest_coincidence(self, uTC, uCoincidences):
        """Select the coincidence with the highest temporal connection
        among the ones contained in uCoincidences."""
        if len(uCoincidences) == 0:
            raise Exception("all coincidences have been assigned.")
        
        while True:
            k = np.argmax(uTC)
            if k in uCoincidences: return k
            else: uTC[k] = -1

    def top_most_connected(self, uK, uCoincidences, uTAM):
        most_connected = []
        uCoincidences.remove(uK)

        if len(uCoincidences) == 0:
            return most_connected
        
        connection_levels = uTAM[uK,:]
        
        while len(most_connected) < self.top_neighbours and len(uCoincidences) != 0:
            print connection_levels
            c = np.argmax(connection_levels)
            if c in uCoincidences: 
                most_connected.append(c)
                uCoincidences.remove(c)

            connection_levels[c] = -1
                
        return most_connected


    def cluster(self, uTC, uTAM):
        """Implements the greedy temporal clustering algorithm."""
        P = []
        coincidences = set(range(uTC))
        Omega = []
        
        while len(coincidences) > 0:
            k = self.select_highest_coincidence(uTC, coincidences)
            coincidences.remove(k)
            Omega.append(k)
            
            while pos <= (len(Omega) - 1) and pos < self.max_group_size:
                k = Omega[k]
                Omega.extend(self.top_most_connected(k, coincidences, uTAM))
                pos += 1
                
            ## mark them as assigned
            for i in range(pos):
                coincidences.remove(Omega[i])
            
            P.append(Omega)
            
        return P

    def compute_PCG(uCoincidencePriors, uTemporalGroups):
        """Compute the PCG matrix."""
        pass
    
    def finalize_training(uNode):
        """Finalize a node's traning by computing its temporal groups and its PCG matrix."""
        ## make TAM symmetric
        norm_TAM = utils.make_symmetric(uNode.TAM)
        
        ## normalize the TAM
        norm_TAM = utils.normalize_over_rows(norm_TAM)
        
        ## compute coincidence priors
        coincidence_priors = uNode.seen / uNode.seen.sum()
        
        ## compute the temporal connections
        TC = np.dot(coincidence_priors, norm_TAM)
        
        ## do the temporal clustering
        uNode.temporal_groups = self.cluster(TC, norm_TAM)
        
        ## compute the PCG matrix
        uNode.PCG = self.compute_PCG(coincidence_priors, uNode.temporal_groups)


if __name__ == "__main__":
    p = TemporalPooler(10, 5, 1)
    TC = np.array([1,2,3,4])
    coincidences = set([0,1,2,3])    
    
    ## test select_highest_coincidence
    # print p.select_highest_coincidence(TC, coincidences)
    
    # coincidences.remove(3)
    # print p.select_highest_coincidence(TC, coincidences)

    # coincidences.remove(1)
    # print p.select_highest_coincidence(TC, coincidences)

    # coincidences.remove(2)
    # print p.select_highest_coincidence(TC, coincidences)

    # coincidences.remove(0)
    # print p.select_highest_coincidence(TC, coincidences)

    ## test top_most_connected
    TAM = np.array([[4,3,2,1],
                    [4,5,6,7],
                    [7,8,9,10],
                    [11,12,13,14]])
    
    print p.top_most_connected(0,range(4), TAM)
    #print p.top_most_connected(1,range(4), TAM)
    #print p.top_most_connected(1,[0,1,2], TAM)
    #print p.top_most_connected(0, [0], TAM)

    

    

    
    
    
    
    
    
