from operator import itemgetter
import numpy as np

def generate_buckets(bitVectors, numBands):
    '''
    Hash each band of numColsPerBand rows into buckets (dictionaries)
    Return a list of dictionaries, one per band
    '''
    bands = np.hsplit(bitVectors, numBands)
    return [hash_band(band) for band in bands]

def hash_band(band):
    '''
    Hash a band of shape [n, numColsPerBand] into buckets.
    Returns a dictionary of 'str representation of numpy array' -> set of indices
    '''
    buckets = {}
    for i, row in enumerate(band):
        key = str(row)
        if key in buckets:
            buckets[key].add(i)
        else:
            buckets[key] = set([i])
    return buckets

def generate_candidates(bucketsOfBands, queryBitVectors, numBands):
    '''
    For each query hash bits, hash each band into a bucket and the indices in
    the bucket becomes candidates for that query
    '''
    candidatesForQueries = []
    for query_num, queryBitVector in enumerate(queryBitVectors):
        bands = np.hsplit(queryBitVector, numBands)
        candidates = set()
        for band, buckets in zip(bands, bucketsOfBands):
            key = str(band)
            if key in buckets:
                candidates.update(buckets[key])
        if not candidates:
            print "No candidate found for query {}".formt(query_num)
        candidatesForQueries.append(list(candidates))
    return candidatesForQueries


class Permutation():
    def __init__(self, sortedData, sortedIdx, permutedIdx):
        # Python list
        self.sortedData = sortedData
        # Python list
        self.sortedIdx = sortedIdx
        # numpy array
        self.permutedIdx = permutedIdx

def generate_permutations(bitVectors, numPermutations):
    '''
    Generate a list of numPermutations number of Permutation objects (hashes)
    '''
    permutations = []
    n, b = bitVectors.shape
    for _ in range(numPermutations):
        permutedIdx = np.random.permutation(b)
        permutedVectors = bitVectors[:, permutedIdx]
        # [(0, [0, 0, 0]), (2, [0, 0, 1]), (1, [0, 1, 0]), (3, [1, 0, 0]), (5, [1, 0, 1]), (4, [1, 1, 0]), ...]
        sortedPermutedVectorsWithIndex = sorted(enumerate(permutedVectors.tolist()), key=itemgetter(1))
        # [(0, 2, 1, 3, 5, 4, ...), ([0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], ...)]
        sortedIdx, sortedData = zip(*sortedPermutedVectorsWithIndex)
        permutations.append(Permutation(sortedData, sortedIdx, permutedIdx))
    return permutations

def lookup(permutations, query):
    '''
    Perform lookup to find candidate nearest neighbors to each query

    :return list of lists, each inner list has candidates for each query
    '''
    numPermutations = len(permutations)
    candidates = []
    for query_num, q in enumerate(query):
        uniqueIndices = set()
        for permutation in permutations:
            permutedQuery = [q[i] for i in permutation.permutedIdx]
            try:
                idx = permutation.sortedData.index(permutedQuery)
                uniqueIndices.update(permutation.sortedIdx[max(idx - 2*numPermutations, 0):min(idx + 2*numPermutations, len(permutation.sortedIdx))])
            except ValueError:
                print "query {} not found in hash table!!!".format(query_num)
                break

        candidates.append(list(uniqueIndices))
    return candidates
