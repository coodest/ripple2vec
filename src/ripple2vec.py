# -*- coding: utf-8 -*-

import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import Manager
from time import time
from collections import deque

from utils import *
from algorithms import *
from algorithms_distances import *
# import graph
from topk import ThresholdTopk


class Graph():
	def __init__(self, g, is_directed, workers, untilLayer = None,method=0,ripple=True):

		logging.info(" - Converting graph to dict...")
		self.G = g.gToDict()
		self.weight = g.weighted
		logging.info("Graph converted.")

		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.is_directed = is_directed
		self.workers = workers
		self.calcUntilLayer = untilLayer
		self.iniLayerZero = True
		self.mDgree = self.max_degree()
		self.method = method
		self.ripple = ripple
		logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
		logging.info('Graph - Number of edges: {}'.format(self.num_edges))


	def max_degree(self):
		mDgree = 0
		for node in self.G.keys():
			if(len(self.G[node])>mDgree):
				mDgree = len(self.G[node])
		return mDgree


	def cal_layer(self):
		rippleList = restoreVariableFromDisk('RippleList')
		self.node_layer = {}
		self.max_depth = 0
		for node in rippleList.keys():
			depth = len(rippleList[node])
			self.node_layer[node] = depth
			if(self.max_depth<depth):
				self.max_depth = depth

	def preprocess_neighbors_with_bfs(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs,self.G,self.weight,self.workers,self.calcUntilLayer,self.mDgree,self.ripple)
			
			self.max_depth = job.result()
			if(self.method==2):
				if(self.max_depth<100):
					if(self.max_depth>1):
						self.method = 100/float(self.max_depth)
					else:
						self.method = 100

		return


	def preprocess_degree_lists(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(preprocess_degreeLists)
			
			job.result()

		return

	def contexLayers(self):
		rippleList = restoreVariableFromDisk('RippleList')
		ripples = dict()
		hitting_k_list = list()
		vertices = rippleList.keys()
		parts = self.workers
		chunks = partition(vertices,parts)
		futures = {}
		
		t0 = time()
		with ProcessPoolExecutor(max_workers = self.workers) as executor:
			part = 1
			for c in chunks:
				ripple_part = {}
				for key in c:
					ripple_part[key] = rippleList[key]
				logging.info("Executing sort part {}...".format(part))
				job = executor.submit(splitHittingTime,c,ripple_part)
				futures[job] = part
				part += 1


			logging.info("Receiving results of sort...")
			for job in as_completed(futures):
				k_hitting_list = job.result()
				for i in range(len(k_hitting_list)):
					if(i > len(hitting_k_list)-1):
						hitting_k_list.append(k_hitting_list[i])
					else:
						hitting_k_list[i].extend(k_hitting_list[i])
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		
		#saveVariableOnDisk(hitting_k_list,"hitting-k")
		if(self.workers>len(hitting_k_list)):
			w = len(hitting_k_list)
		else:
			w = self.workers
		chunks = partition(range(len(hitting_k_list)),w)
		futures = {}
		sorted_hitting_list = [0]*len(hitting_k_list)
		hitting_map_list = [0]*len(hitting_k_list)
		with ProcessPoolExecutor(max_workers = w) as executor:
			part = 1
			for c in chunks:
				logging.info("Executing sort part {}...".format(part))
				job = executor.submit(sortHittingTime,hitting_k_list[c[0]:c[len(c)-1]+1],c)
				futures[job] = part
				part += 1


			logging.info("Receiving results of sort...")
			for job in as_completed(futures):
				sorted_hitting_k,map_k_hitting,c = job.result()
				sorted_hitting_list[c[0]:c[len(c)-1]+1] = sorted_hitting_k
				hitting_map_list[c[0]:c[len(c)-1]+1] = map_k_hitting
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		chunks = partition(vertices,parts)
		futures = {}
		k_neighbors = dict()
		t2 = time()
		# saveVariableOnDisk(sorted_hitting_list,"sorted-hitting")
		# saveVariableOnDisk(hitting_map_list,"hitting-map")
		# sorted_hitting_list = restoreVariableFromDisk("sorted-hitting")
		# hitting_map_list = restoreVariableFromDisk("hitting-map")
		with ProcessPoolExecutor(max_workers = self.workers) as executor:
			part = 1
			for c in chunks:
				logging.info("Executing sort part {}...".format(part))
				job = executor.submit(ThresholdTopk,c,self.G,self.node_layer,sorted_hitting_list,hitting_map_list,part,self.max_depth,self.method)
				futures[job] = part
				part += 1


			logging.info("Receiving results of sort...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		t1 = time()
		logging.info('OPT1 cost. Time: {}s'.format((t1-t0)))
		logging.info('OPT1 cost2. Time: {}s'.format((t1-t2)))

	def create_vectors(self):
		logging.info("Creating degree vectors...")
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]
		logging.info("Degree vectors created.")
		logging.info("Saving degree vectors...")
		saveVariableOnDisk(degrees,'degrees_vector')


	def calc_distances_all_vertices(self):
    
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}

		count_calc = 0

		vertices = list(reversed(sorted(self.G.keys())))

		logging.info("Recovering compactDegreeList from disk...")
		rippleList = restoreVariableFromDisk('RippleList')
		

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()
		
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				list_v = []
				for v in c:
					list_v.append([vd for vd in rippleList.keys() if vd > v])
				job = executor.submit(calc_distances_ripple, c, list_v, rippleList,part,self.max_depth,self.method)
				futures[job] = part
				part += 1


			logging.info("Receiving results...")

			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		
		logging.info('Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))
		
		return




	def calc_distances_neighbors(self):
        
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}

		count_calc = 0

		vertices = list(reversed(sorted(self.G.keys())))

		logging.info("Recovering compactDegreeList from disk...")
		rippleList = restoreVariableFromDisk('RippleList')
		

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()
		
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				list_v = []
				for v in c:
					list_v.append([vd for vd in rippleList.keys() if vd > v])
				job = executor.submit(calc_distances_ripple, c, list_v, rippleList,part,self.max_depth,self.method)
				futures[job] = part
				part += 1


			logging.info("Receiving results...")

			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		
		logging.info('Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))
		
		return

	def calc_distances(self):
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		G = self.G
		vertices = G.keys()

		parts = self.workers
		chunks = partition(vertices,parts)


		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			logging.info("Split degree List...")
			part = 1
			for c in chunks:
				job = executor.submit(splitRippleList,part,c,G)
				job.result()
				logging.info("RippleList {} completed.".format(part))
				part += 1

		
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances, part,self.max_depth,self.method)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))


		return

	def consolide_distances(self):

		distances = {}

		parts = self.workers
		for part in range(1,parts + 1):
			d = restoreVariableFromDisk('distances-'+str(part))
			preprocess_consolides_distances(distances)
			distances.update(d)


		preprocess_consolides_distances(distances)
		saveVariableOnDisk(distances,'distances')


	def create_distances_network(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_distances_network,self.workers)

			job.result()

		return

	def preprocess_parameters_random_walk(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_parameters_random_walk,self.workers)

			job.result()

		return


	def simulate_walks(self,num_walks,walk_length):

		# for large graphs, it is serially executed, because of memory use.
		if(len(self.G) > 500000):
			generate_random_walks_large_graphs(num_walks,walk_length,self.workers,list(self.G.keys()),self.node_layer,self.iniLayerZero)

		else:
			generate_random_walks(num_walks,walk_length,self.workers,list(self.G.keys()),self.node_layer,self.iniLayerZero)


		return	





		

      	


