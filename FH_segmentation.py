import numpy as np


### A class representing a partitioned set

class DisjointSet:

    def __init__(self, size):
        self.number_subsets = size
        self.elements = np.empty(shape=(size, 2), dtype=int)
        for i in range(size):
            self.elements[i, 0] = 1
            self.elements[i, 1] = i

    def find(self, a):
        b = int(a)
        while b != self.elements[b, 1]:
            b = self.elements[b, 1]
        self.elements[a, 1] = b
        return b

    def subset_size(self, a):
        return self.elements[self.find(a), 0]

    def merge(self, a, b):
        #if self.find(a) != self.find(b):
        # Add this if statement if the merge method may be called
        # with a and b in the same component
        self.number_subsets -= 1
        self.elements[b, 1] = a
        self.elements[a, 0] += self.elements[b, 0]


### An implementation Felzenszwalb and Huttenlocher's graph segmentation

def graph_segmentation(size, edges, k):
    # Almost identical to soumik12345's implementation on GitHub
    number_edges = len(edges)
    edges[0 : number_edges, :] = edges[edges[0 : number_edges, 2].argsort()]
    u = DisjointSet(size)
    internal_differences = np.zeros(size, dtype = int)
    kept_edges = []
    while len(edges) > 0:
        edge = edges[0]
        a, b = u.find(edge[0]), u.find(edge[1])
        if a != b:
            if (edge[2] <= internal_differences[a] + k/u.subset_size(a)
                    and (edge[2] <= internal_differences[b] + k/u.subset_size(b))):
                u.merge(a, b)
                internal_differences[u.find(a)] = edge[2]
            else:
                edge[0], edge[1] = a, b
                kept_edges.append(edge)
        edges = edges[1:]
    return u, np.array(kept_edges, dtype = int)

def merge_small_components(u, edges, min_size):
    # Merge the components of size < min_size connected by edges with small dissimilarity
    kept_edges = []
    while len(edges) > 0:
        edge = edges[0]
        a, b = u.find(edge[0]), u.find(edge[1])
        if a != b:
            if u.subset_size(a) < min_size or u.subset_size(b) < min_size:
                u.merge(a, b)
            else:
                edge[0], edge[1] = a, b
                kept_edges.append(edge)
        edges = edges[1:]
    return u, kept_edges