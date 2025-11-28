#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################################
# MIT License
#
# Copyright (c) 2016 Leonardo Filipe Rodrigues Ribeiro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
#############################################################################################

# import logging
# import sys
# import math
from io import open
# from os import path
from time import time
# from glob import glob
# from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
# from multiprocessing import cpu_count
# import random
# from random import shuffle
# from itertools import product,permutations
# import collections

# from concurrent.futures import ProcessPoolExecutor

# from multiprocessing import Pool
# from multiprocessing import cpu_count

# import numpy as np
# import operator


class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()


  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    t1 = time()
    #logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    return self




  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order() 

  def gToDict(self):
    d = {}
    for k,v in self.items():
      d[k] = v
    return d


def load_edgelist(file_, undirected=True,weighted=False):
  G = Graph()
  G.weighted = dict()
  with open(file_) as f:
    for l in f:
      if(len(l.strip().split()[:2]) > 1):
        x, y = l.strip().split()[:2]
        x = int(x)
        y = int(y)
        G[x].append(y)
        if undirected:
          G[y].append(x)
      else:
        x = l.strip().split()[:2]
        x = int(x[0])
        G[x] = []  
      if(len(l.strip().split()[:2]) > 2 and weighted):
        G.weighted[(x,y)] = l.strip().split()[2]
        if undirected:
          G.weighted[(y,x)] = l.strip().split()[2]
      else:
        G.weighted[(x,y)] = 1
        if undirected:
          G.weighted[(y,x)] = 1
  
  G.make_consistent()
  return G

