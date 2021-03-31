#!/usr/bin/env python3
import torch
import GNNAdvisor as GNNA
import sys
import time
from tqdm import *

class Verification(object):
    def __init__(self, dim, row_pointers, column_index, degrees, partPtr, part2Node, \
                partSize, dimWorker, warpPerBlock):

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.partPtr = partPtr
        self.part2Node = part2Node

        self.warpPerBlock = warpPerBlock      
        self.partSize = partSize
        self.dimWorker = dimWorker

        self.num_nodes = len(row_pointers) - 1
        self.test_embedding = dim
        self.output_embedding = dim

        self.X = torch.ones(self.num_nodes, self.test_embedding)
        self.W = torch.ones(self.test_embedding, self.output_embedding)

        self.result = None
        self.result_ref = None
        
    def reference(self):
        '''
        Compute reference SpMM (neighbor aggregation)
        result on CPU.
        '''
        print("# Compute reference on CPU")
        self.result_ref = torch.zeros_like(self.X)

        for i in range(len(self.row_pointers) - 1):
            for eidx in range(self.row_pointers[i], self.row_pointers[i+1]):
                eid = self.column_index[eidx]
                for d in range(len(self.result_ref[0])):
                    self.result_ref[i][d] +=  self.X[eid][d]
        print(self.result_ref)

    def compute(self):
        '''
        Compute SpMM (neighbor aggregation)
        result on GPU.
        '''
        print("# Compute result on GPU")
        X = self.X.cuda()
        self.result = GNNA.SAG(X, self.row_pointers, self.column_index, self.degrees,\
                                    self.partPtr, self.part2Node, self.partSize, self.dimWorker, self.warpPerBlock)
        # print(self.result)


    def compare(self):
        if self.result_ref is None or self.result is None:
            raise ValueError("MUST compute result and result reference (CPU) first!!")

        if torch.all(torch.eq(self.result_ref, self.result.cpu())):
            print("Verification PASSED")
        else:
            print("Verification FAILED")

    def profile_spmm(self, round=200):
        X = self.X.cuda()
        print("SpMM profiling size: N: {}, N: {}, K: {}".format(X.size(0), X.size(0), X.size(1)))
        # dry run
        for _ in range(10):
            self.result = GNNA.SAG(X, self.row_pointers, self.column_index, self.degrees,\
                                    self.partPtr, self.part2Node, self.partSize, self.dimWorker, self.warpPerBlock)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in tqdm(range(round)):
            self.result = GNNA.SAG(X, self.row_pointers, self.column_index, self.degrees,\
                            self.partPtr, self.part2Node, self.partSize, self.dimWorker, self.warpPerBlock)
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> SpMM profiling avg (ms): {:.3f}".format(dur*1e3/round))