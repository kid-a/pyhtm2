## global import ---------------------------------------------------------------
import numpy as np

## local import ----------------------------------------------------------------
import utils


## -----------------------------------------------------------------------------
## tc2dict function
## -----------------------------------------------------------------------------
def tc2dict(TC):
    """Given the TC vector, returns a dict of <connectivity degree-node> pairs."""
    d = {}
    for i in range(len(TC)): d[TC[i]] = []
    for i in range(len(TC)): d[TC[i]].append(i)
    return d


## -----------------------------------------------------------------------------
## remove_tc function
## -----------------------------------------------------------------------------
def remove_tc(node, TC):
    """Remove the specified node from the given TC dict."""
    for i in TC.keys():
        l = TC[i]
        try: 
            l.remove(node)
            if len(l) == 0: del TC[i]
            break

        except: continue


## -----------------------------------------------------------------------------
## tam2adjlist
## -----------------------------------------------------------------------------
def tam2adjlist(TAM):
    """Converts a TAM into the correspondent adjacency list."""
    adjlist = {}
    (rows, cols) = TAM.shape

    for i in range(rows):
        adjlist[i] = []
        for j in range(cols):
            if i == j: continue ## skip auto-connections
            if TAM[i,j] > 0:
                adjlist[i].append((j, TAM[i,j]))
                
    return adjlist


## -----------------------------------------------------------------------------
## remove_adjlist
## -----------------------------------------------------------------------------
def remove_adjlist(node, adjlist):
    """Remove the specified node from the given adjacency list."""
    del adjlist[node]

    for n in adjlist:
        for (t,w) in adjlist[n]:
            if t == node:
                adjlist[n].remove((t,w))    

## -----------------------------------------------------------------------------
## TemporalPooler Class
## -----------------------------------------------------------------------------
class TemporalPooler(object):
    def greedy_temporal_clustering(self, uTC, uTAM, uParams):
        """Implements the greedy temporal clustering algorithm."""
        #graph = tam2adjlist(uTAM)
        (coinc_count, coinc_count) = uTAM.shape
        tc = tc2dict(uTC)
        partition = []
        assigned = []

        max_group_size = uParams['max_group_size']
        
        while len(assigned) < coinc_count:
            (k, tc) = self.pop_highest_coincidence(tc)
            omega = set([k])
            unprocessed = [k]
            processed = []
            
            # print "k:", k
            # print "unprocessed:", unprocessed
            # print "processed:", processed
            # print "omega:", omega
            # print
                        
            while len(unprocessed) > 0 and len(processed) < max_group_size:
                k = unprocessed[0] ## pick an unprocessed node
                most_connected = self.top_most_connected(uTAM, k, assigned, uParams)
                omega = omega.union(most_connected)
                
                processed.append(k)
                unprocessed.remove(k)
                for n in most_connected:
                    if n not in unprocessed and n not in processed:
                        unprocessed.append(n)
                    
                #unprocessed.extend(most_connected)

                #unprocessed = list(set(unprocessed).difference(set(processed)))
                ## maybe here the order is not respected
                # print "k:", k
                # print "most_connected:", most_connected
                # print "unprocessed:", unprocessed
                # print "processed:", processed
                # print "omega:", omega
                # print
                           
            for n in omega:
                assigned.append(n)
                #remove_adjlist(n, graph)
                #uTAM.mask[n,n] = True
                remove_tc(n, tc)

            #np.ma.mask_cols(uTam)

            partition.append(list(omega))
            
        print sum(assigned)
        
        return partition
    
    def pop_highest_coincidence(self, uTC):
        """Get the coincidence with the highest temporal connectivity and 
        remove it from the given TC dict."""
        if len(uTC) == 0: return (None, None)
        else:
            m = max(uTC)
            k = uTC[m][0]
            uTC[m].remove(k)
            
            if len(uTC[m]) == 0: del uTC[m]
            
            return (k, uTC)

    def top_most_connected(self, uTAM, uSource, uAssigned, uParams):
        """Returns the top-most-connected nodes to the given source."""
        print "processing node", uSource
        most_connected = []
        edge_weights = uTAM[uSource]
        # edge_indices = (np.ma.nonzero(uTAM[uSource])[0]).tolist()
        # edge_weights = (uTAM[np.ma.nonzero(uTAM)]).tolist()

        # adjlist = zip(edge_indices, edge_weights)
        adjlist = edge_weights.tolist()
        indices = range(len(adjlist))

        # # del adjlist[uSource]
        # # indices.remove(uSource)

        # # for node in uAssigned:
        # #     del adjlist[node]
        # #     #adjlist.remove(node)
        # #     indices.remove(node)

        adjlist = zip(indices, adjlist)
        adjlist = [(x,y) for (x,y) in adjlist if x not in uAssigned]
        adjlist = [(x,y) for (x,y) in adjlist if x != uSource]

        #adjlist = filter(lambda x : if x[0] == Source return
    
        # for j in range(len(edge_weights)):
        #     if uSource == j: continue
        #     if j in uAssigned: continue
        #     if edge_weights[j] > 0:
        #         adjlist.append((j, edge_weights[j]))
        
        top_neighbours = uParams['top_neighbours']

        sorted_adjlist = sorted(adjlist, key=lambda x : x[1], reverse=True)
        
        if len(adjlist) <= top_neighbours:
            return map(lambda x : x[0], sorted_adjlist)

        else:
            return map(lambda x : x[0], sorted_adjlist[:top_neighbours])

    def compute_PCG(self, uCoincidencePriors, uTemporalGroups):
        """Compute the PCG matrix."""
        PCG = np.zeros((len(uCoincidencePriors), len(uTemporalGroups)))
        
        for i in range(len(uCoincidencePriors)):
            for j in range(len(uTemporalGroups)):
                if i in uTemporalGroups[j]:
                    PCG[i,j] = uCoincidencePriors[i]
                    
        return utils.normalize_over_cols(PCG)

    def finalize(self, uNodeState):
        """Finalize a node's traning by computing its temporal 
        groups and its PCG matrix."""
        TAM = uNodeState['TAM']
        seen = uNodeState['seen']
        
        params = {}
        params['top_neighbours'] = uNodeState['top_neighbours']
        params['max_group_size'] = uNodeState['max_group_size']
        
        ## check whether the last coincidence added was a temporal gap
        ## and fix the TAM
        ## if np.all(TAM[-1] == 0): TAM[-1][-1] = 1
                
        ## make TAM symmetric
        norm_TAM = utils.make_symmetric(TAM)
        
        ## normalize the TAM
        norm_TAM = np.nan_to_num(utils.normalize_over_rows(norm_TAM))
        
        # ## transform TAM into a masked array
        # norm_TAM = np.ma.masked_array(norm_TAM)
        # norm_TAM.mask = np.ma.make_mask_none(norm_TAM.shape)
        
        ## compute coincidence priors
        coincidence_priors = np.array(seen, dtype=np.float32) / float(seen.sum())
        
        ## compute the temporal connections
        TC = np.dot(coincidence_priors, norm_TAM)
        
        ## do the temporal clustering
        temporal_groups = self.greedy_temporal_clustering(TC, norm_TAM, params)
        
        print len(temporal_groups)
                
        ## compute the PCG matrix
        PCG = self.compute_PCG(coincidence_priors, temporal_groups)
        
        uNodeState['temporal_groups'] = temporal_groups
        uNodeState['PCG'] = PCG


if __name__ == "__main__":
    p = TemporalPooler()
    
    TAM = np.array([[0, 0.5, 0, 0.3, 0.6, 0],
                    [0, 0, 0.2, 0, 0, 0],
                    [0, 0.4, 0, 0, 0, 0],
                    [0, 0.1, 0, 0, 0.4, 0.3],
                    [0, 0, 0, 0, 0, 0.4],
                    [0, 0, 0.1, 0.2, 0, 0]])

    TC = np.array([0.4, 0.3, 0.5, 0.2, 0.6, 0.7])
    
    part = p.greedy_temporal_clustering(TC, TAM, {'top_neighbours' : 2,
                                                  'max_group_size' : 2})
    
    print part
    

