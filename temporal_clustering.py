import numpy as np
import copy
import utils
import network


def tc2dict(TC):
    """Given the TC vector, returns a dict of <connectivity-node> pairs."""
    d = {}
    for i in range(len(TC)): d[TC[i]] = []
    for i in range(len(TC)): d[TC[i]].append(i)
    return d


def remove_tc(node, TC):
    for i in TC.keys():
        l = TC[i]
        try: 
            l.remove(node)
            if len(l) == 0: del TC[i]
            break

        except: continue


def tam2adjlist(TAM):
    adjlist = {}
    (rows, cols) = TAM.shape

    for i in range(rows):
        adjlist[i] = []
        for j in range(cols):
            if i == j: continue ## skip auto-connections
            if TAM[i,j] > 0:
                adjlist[i].append((j, TAM[i,j]))
                
    return adjlist


def remove_adjlist(node, adjlist):
    del adjlist[node]

    for n in adjlist:
        for (t,w) in adjlist[n]:
            if t == node:
                adjlist[n].remove((t,w))
    

class TemporalPooler(object):
    def __init__(self, uMaxGroupSize, uMinGroupSize, uTopNeighbours):
        self.max_group_size = uMaxGroupSize
        self.min_group_size = uMinGroupSize ##!FIXME it is not used
        self.top_neighbours = uTopNeighbours

    def greedy_temporal_clustering(self, uTC, uTAM):
        """Implements the greedy temporal clustering algorithm."""
        graph = tam2adjlist(uTAM)
        tc = tc2dict(uTC)
        partition = []
        
        while len(graph) > 0:
            (k, tc) = self.pop_highest_coincidence(tc)
            omega = set([k])
            unprocessed = [k]
            processed = []
            
            while len(unprocessed) > 0 and len(processed) < self.max_group_size:
                k = unprocessed[0] ## pick an unprocessed node
                most_connected = self.top_most_connected(graph, k)
                omega = omega.union(most_connected)
                
                processed.append(k)
                unprocessed.remove(k)
                unprocessed.extend(most_connected)
                unprocessed = list(set(unprocessed).difference(set(processed)))
                
                # if len(unprocessed) == 0 and \
                #         len(omega) < self.min_group_size and \
                #         len(tc) != 0:

                #     (k, tc) = self.pop_highest_coincidence(tc)
                #     omega.add(k)
                #     unprocessed = [k]
                #     continue
                
            for n in omega:
                remove_adjlist(n, graph)
                remove_tc(n, tc)
            
            partition.append(list(omega))
        
        return partition
    
    def pop_highest_coincidence(self, uTC):
        if len(uTC) == 0: return (None, None)
        else:
            m = max(uTC)
            k = uTC[m][0]
            uTC[m].remove(k)
            
            if len(uTC[m]) == 0: del uTC[m]
            
            return (k, uTC)

    def top_most_connected(self, uGraph, uSource):
        most_connected = []
        neighbours = uGraph[uSource]
        
        if len(neighbours) < self.top_neighbours:
            return map(lambda x : x[0], neighbours)
        
        else:
            sorted_neighbours = sorted(neighbours, key=lambda x : x[1])
            return map(lambda x : x[0], neighbours[:self.top_neighbours])

    def compute_PCG(self, uCoincidencePriors, uTemporalGroups):
        """Compute the PCG matrix."""
        PCG = np.zeros((len(uCoincidencePriors), len(uTemporalGroups)))
        
        for i in range(len(uCoincidencePriors)):
            for j in range(len(uTemporalGroups)):
                if i in uTemporalGroups[j]:
                    PCG[i,j] = uCoincidencePriors[i]
                    
        return utils.normalize_over_cols(PCG)

    def finalize_training(self, uNode):
        """Finalize a node's traning by computing its temporal groups and its PCG matrix."""
        ## make TAM symmetric
        norm_TAM = utils.make_symmetric(uNode.TAM)
        
        ## normalize the TAM
        norm_TAM = utils.normalize_over_rows(norm_TAM)
        
        ## compute coincidence priors
        coincidence_priors = uNode.seen / float(uNode.seen.sum())
        
        ## compute the temporal connections
        TC = np.dot(coincidence_priors, norm_TAM)
        
        ## do the temporal clustering
        uNode.temporal_groups = self.greedy_temporal_clustering(TC, norm_TAM)
                
        ## compute the PCG matrix
        uNode.PCG = self.compute_PCG(coincidence_priors, uNode.temporal_groups)


if __name__ == "__main__":
    p = TemporalPooler(5, 3, 3)
    TC = np.array([1,2,3,3])
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
    
    #print p.top_most_connected(0,[1,2,3], TAM)
    #print p.top_most_connected(1,[0,2,3], TAM)
    #print p.top_most_connected(1,[0,2], TAM)
    #print p.top_most_connected(0, [1], TAM)
    #print p.top_most_connected(0, [], TAM)

    ## test cluster
    TC = np.random.rand(1, 100)
    TAM = np.random.rand(100, 100)

    cluster = p.greedy_temporal_clustering(TC[0], TAM)
    s = 0
    for c in cluster:
        s += len(c)
        print len(c)
        
    print "Total coinc:", s

    for c in cluster:
        print c

    ## testing tam2adjlist and remove
    # graph = tam2adjlist(TAM)
    # print graph
    
    # remove(2, graph)
    # print graph
    
    ## testing p.pop_highest_coincidence
    # print tc2dict(TC)
    # (k, TC) = p.pop_highest_coincidence(tc2dict(TC))
    # print k, TC
    # (k, TC) = p.pop_highest_coincidence(TC)
    # print k, TC
    # (k, TC) = p.pop_highest_coincidence(TC)
    # print k, TC
    # (k, TC) = p.pop_highest_coincidence(TC)
    # print k, TC
    # print p.pop_highest_coincidence(TC)

    ## testing remove_tc
    # print tc2dict(TC)
    # TC = tc2dict(TC)
    # remove_tc(1,TC)
    # print TC
    # remove_tc(2,TC)
    # print TC
    # remove_tc(3,TC)
    # print TC
    # remove_tc(0,TC)
    # print TC
    # remove_tc(0,TC)
    # print TC

    ## testing top_most_connected
    # graph = tam2adjlist(TAM)
    # print graph
    # print 
    # print p.top_most_connected(graph, 0)



    

    

    
    
    
    
    
    
