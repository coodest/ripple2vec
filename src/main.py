#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
# import numpy as np
import ripple2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# from time import time

import graph


def parse_args():
	'''
	Parses the ripple2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run ripple2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=4,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.add_argument('--log', default='ripple2vec.log',help='log file',dest='log')
	parser.set_defaults(directed=False)
	parser.add_argument('--method', default=0, type=int, help='the method to calculate distance')

	parser.set_defaults(directed=False)

	parser.add_argument('--OPT1', default=True, type=bool,
                      help='optimization 1.If set to True, the network is constructed without calculating the distances of all node pairs and instead calculates the distances of the nearest points of each point.')
	parser.add_argument('--OPT2', default=True, type=bool,
                      help='optimization 2.If set to True, the maximum number of layers to calculate the hitting time is until_layer.Otherwise, all layers are calculated.')
	parser.add_argument('--OPT3', default=False, type=bool,
                      help='optimization 3.If set to True, random walk will begin from the last layer,If False,random walk will begin from the first layer.')
	parser.add_argument('--OPT4', default=True, type=bool,
                      help='optimization 4.If set to True, the hitting time will be calculated by abstracting the neighborhood into a graph of five vertices.')
	return parser.parse_args()

def read_graph():
	'''
	Reads the input network.
	'''
	logging.info(" - Loading graph...")
	G = graph.load_edgelist(args.input,undirected=True,weighted=False)
	logging.info(" - Graph loaded.")
	return G

def learn_embeddings():
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	walks = LineSentence('random_walks.txt')
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	logging.info("Representations created.")
	
	return

def exec_ripple2vec(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	logging.basicConfig(filename=args.log,filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')
	if(args.OPT2):
		until_layer = args.until_layer
	else:
		until_layer = None

	G = read_graph()
	G = ripple2vec.Graph(G, args.directed, args.workers, untilLayer = until_layer, method = args.method,ripple=args.OPT4)
	G.preprocess_neighbors_with_bfs()
	G.cal_layer()
	if(args.OPT3):
		G.iniLayerZero = False
	if(args.OPT1):
		G.contexLayers()
	else:
		G.calc_distances_all_vertices()
	

	G.create_distances_network()
	G.preprocess_parameters_random_walk()

	G.simulate_walks(args.num_walks, args.walk_length)


	return G

def main(args):

	G = exec_ripple2vec(args)

	learn_embeddings()


if __name__ == "__main__":
	args = parse_args()
	main(args)

