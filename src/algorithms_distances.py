# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,logging
# from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
# from collections import defaultdict
from utils import *
from heapq import *
import os

limiteDist = 20

#Get the knd ripple vector,the format is{vertex:{0:sequence_0,...k:sequence_k}}
def getRippleListsVertices(g,weight,vertices,calcUntilLayer,mDgree,ripple):
    rippleList = {}
    max_depth=0

    for v in vertices:
        if(ripple):
            rippleList[v] = getRippleList2(g,weight,v,calcUntilLayer,mDgree)
        else:
            rippleList[v] = getRippleList(g,weight,v,calcUntilLayer,mDgree)
        if(len(rippleList[v])>max_depth):
            max_depth = len(rippleList[v])
    return rippleList,max_depth

def kndTransform(g,weight,node,v_nodes,k_neighbors):
    node_neighbors = g[node]
    weight_sum = 0
    neighbor_sum = 0
    for neighbor in node_neighbors:
        weight_sum += weight[(node,neighbor)]
    dim = len(k_neighbors)+1
    transform = np.zeros(dim)
    for i in range(dim-1):
        if(k_neighbors[i] in node_neighbors):
            neighbor_sum+=weight[(node,k_neighbors[i])]
            transform[i+1] = float(weight[(node,k_neighbors[i])])/weight_sum
    for v in v_nodes:
        if(v in node_neighbors):
            neighbor_sum+= weight[(node,v)]
            transform[0] = transform[0]+weight[(node,v)]
    transform[0] = float(transform[0])/weight_sum
    away = True
    if(abs(weight_sum-neighbor_sum)<1e-6):
        away = False
    return transform,away

    
def hitting_time_cal(B,N,l):
    dim = B.shape[0]
    power = np.mat(np.ones(dim)).T
    result = np.mat(np.ones(dim)).T
    for i in range(N):
        power = B*power
        result = result+power
        error = np.linalg.norm(power)
        if(error<l):
            break
    return result


def getRippleList(g,weight, root, calcUntilLayer,mDgree):
    
    t0 = time()

    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    d = math.log(float(len(g.keys())),mDgree)/3
    if(d<2):
        d=2
    #calcUntilLayer = d

    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1
    
    v_nodes = {root}
    k_neighbors = dict()
    vector = list()
    vector.append(len(g[root]))
    sum_weight = 0
    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1


        for v in g[vertex]:
            if not(v in v_nodes):
                if(v in k_neighbors):
                    k_neighbors[v] = k_neighbors[v]+weight[(vertex,v)]
                else:
                    k_neighbors[v] = weight[(vertex,v)]
                sum_weight+=weight[(vertex,v)]
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  


        if(timeToDepthIncrease == 0):
            for node in k_neighbors.keys():
                k_neighbors[node] = float(k_neighbors[node])/sum_weight
            sum_weight = 0
            if not(len(v_nodes)+len(k_neighbors)==len(g.keys())) and len(queue)>0:
                Away = False
                k_neighbors_nodes = list(k_neighbors.keys())
                dim = len(k_neighbors_nodes)+1
                B = np.mat(np.zeros((dim,dim)))
                for i in range(dim-1):
                    B[0,i+1] = k_neighbors[k_neighbors_nodes[i]]
                for i in range(dim-1):
                    B[i+1,:],away = kndTransform(g,weight,k_neighbors_nodes[i],v_nodes,k_neighbors_nodes)
                    if(away==True):
                        Away = True
                if(Away):
                    #vector.append((np.eye(dim)-B).I.dot(np.ones(dim))[0,0])
                    vector.append(hitting_time_cal(B,100,0.1)[0,0])
                
            v_nodes.update(k_neighbors.keys())
            k_neighbors = dict()
            

            if(calcUntilLayer == depth):
                break


            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0
        if(calcUntilLayer == depth):
            break
    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))


    return vector

def getHittingTime(g,weight,k_neighbors,v_nodes):
    B = np.mat(np.zeros((4,4)))
    k_neighbors_2 = dict()
    k_neighbors_weight = float(0)
    node_weights = dict()
    v1 = set()
    v2 = set()
    v3 = set()
    for node in k_neighbors.keys():
        flag1 = False
        flag2 = False
        for neighbor in g[node]:
            if(neighbor in k_neighbors):
                flag1 = True
            elif not(neighbor in v_nodes):
                flag2 = True
                break
        if(flag2):
            v3.add(node)
        elif(flag1):
            v2.add(node)
        else:
            v1.add(node)
    if(len(v3)==0):
        return -1,k_neighbors_2
    if(len(v1)>0):
        for node in v1:
            B[0,1] +=k_neighbors[node]
        B[1,0] = 1
    sum_weight = 0
    if(len(v2)>0):
        for node in v2:
            B[0,2] += k_neighbors[node]
            for neighbor in g[node]:
                w = weight[(node,neighbor)]
                sum_weight += w
                if(neighbor in v_nodes):
                    B[2,0] += w
                elif(neighbor in v2):
                    B[2,2] += w
                elif(neighbor in v3):
                    B[2,3] += w
        for i in range(4):
            B[2,i] = B[2,i]/float(sum_weight)
    sum_weight=0
    if(len(v3)>0):
        for node in v3:
            B[0,3] += k_neighbors[node]
            for neighbor in g[node]:
                w = weight[(node,neighbor)]
                sum_weight += w
                if(neighbor in v_nodes):
                    B[3,0] += w
                elif(neighbor in v2):
                    B[3,2] += w
                elif(neighbor in v3):
                    B[3,3] += w
                else:
                    k_neighbors_weight += w
                    if(neighbor in k_neighbors_2):
                        k_neighbors_2[neighbor] = k_neighbors_2[neighbor]+w
                    else:
                        k_neighbors_2[neighbor] = w
        for neighbor in k_neighbors_2.keys():
            k_neighbors_2[neighbor] = k_neighbors_2[neighbor]/k_neighbors_weight
        for i in range(4):
            B[3,i] = B[3,i]/float(sum_weight)
    return (np.eye(4)-B).I.dot(np.ones(4))[0,0],k_neighbors_2


        
                


def getRippleList2(g,weight, root, calcUntilLayer,mDgree):
    t0 = time()

    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    # d = math.log(float(len(g.keys())),mDgree)/3
    # if(d<2):
    #     d=2
    #calcUntilLayer = d

    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1
    
    v_nodes = {root}
    k_neighbors = dict()
    vector = list()
    vector.append(len(g[root]))
    sum_weight = float(0)
    for vertex in g[root]:
        sum_weight+=weight[(root,vertex)]
        if(vertex in k_neighbors):
            k_neighbors[vertex] = k_neighbors[vertex]+weight[(root,vertex)]
        else:
            k_neighbors[vertex] = weight[(root,vertex)]
    for vertex in k_neighbors.keys():
        k_neighbors[vertex] = k_neighbors[vertex]/sum_weight
    
    while(len(k_neighbors)>0):
        hitting,k_neighbors_2 = getHittingTime(g,weight,k_neighbors,v_nodes)
        if(hitting!=-1):
            vector.append(hitting)
            v_nodes.update(k_neighbors)
            k_neighbors = k_neighbors_2
        else:
            break
        
        depth += 1
        if(calcUntilLayer == depth):
                break

    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))
    return vector

def cost(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)

def cost_min(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * min(a[1],b[1])


def cost_max(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * max(a[1],b[1])

def preprocess_degreeLists():

    logging.info("Recovering degreeList from disk...")
    degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Creating compactDegreeList...")

    dList = {}
    dFrequency = {}
    for v,layers in degreeList.items():
        dFrequency[v] = {}
        for layer,degreeListLayer in layers.items():
            dFrequency[v][layer] = {}
            for degree in degreeListLayer:
                if(degree not in dFrequency[v][layer]):
                    dFrequency[v][layer][degree] = 0
                dFrequency[v][layer][degree] += 1
    for v,layers in dFrequency.items():
        dList[v] = {}
        for layer,frequencyList in layers.items():
            list_d = []
            for degree,freq in frequencyList.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d,dtype='float')

    logging.info("compactDegreeList created!")

    saveVariableOnDisk(dList,'compactDegreeList')

def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            
            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)

def compare(a):
    return a[1]

def splitHittingTime(c,rippleList):
    hitting_time_k = []
    for vertex in rippleList.keys():
        hitting_list = rippleList[vertex]
        for i in range(len(hitting_list)):
            if(i>len(hitting_time_k)-1):
                hitting_time_k.append([(vertex,hitting_list[i])])
            else:
                hitting_time_k[i].append((vertex,hitting_list[i]))
    return hitting_time_k

def sortHittingTime(hitting_time_list,c):
    sorted_hitting_time = list()
    for hitting_time_k in hitting_time_list:
        sorted_hitting_time.append(sorted(hitting_time_k,key=compare))
    map_k_hitting = list()
    for i in range(len(c)):
        map_k_hitting.append(dict())
        for j in range(len(sorted_hitting_time[i])):
            map_k_hitting[i][sorted_hitting_time[i][j][0]] = j
    return sorted_hitting_time,map_k_hitting,c

def splitRippleList(part,c,G):

    rippleList = restoreVariableFromDisk('RippleList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')

    rippleListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v,len(G[v]),degrees,a_vertices)
        vertices[v] = nbs
        rippleListsSelected[v] = rippleList[v]
        for n in nbs:
            rippleListsSelected[n] = rippleList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(rippleListsSelected,'split-rippleList-'+str(part))

def algorithms_ripple(distance,d,layer,max_layer,method):
    if(method == 0):
        return distance+d*(max_layer-layer+1)
    if(method == 1):
        return distance+d*(layer+1)
    if(method==3):
        return distance+d
        if(layer<4):
            w = 20
        else:
            w = 20/float(max_layer-4)
        return distance+w*d
    else:
        return distance+d


def ripple_distance(list_v1,list_v2,layer,max_layer,method):
    distance = 0
    for i in range(layer+1):
        if(len(list_v1)<=i):
            hitting_time_v1 = 0
        else:
            hitting_time_v1 = list_v1[i]
        if(len(list_v2)<=i):
            hitting_time_v2 = 0
        else:
            hitting_time_v2 = list_v2[i]
        if(hitting_time_v1 > hitting_time_v2):
            Max = hitting_time_v1
            Min = hitting_time_v2
        else:
            Max = hitting_time_v2
            Min = hitting_time_v1
        if(Max>0):
            #distance = distance+(float(Max)/Min)-1
            distance = algorithms_ripple(distance,(1-float(Min)/Max),i,max_layer,method)
    #distance = distance*distance
    # if(len(list_v1)<=layer):
    #     hitting_time_v1 = 0
    # else:
    #     hitting_time_v1 = list_v1[layer]
    # if(len(list_v2)<=layer):
    #     hitting_time_v2 = 0
    # else:
    #     hitting_time_v2 = list_v2[layer]
    # if(hitting_time_v1 > hitting_time_v2):
    #     Max = hitting_time_v1
    #     Min = hitting_time_v2
    # else:
    #     Max = hitting_time_v2
    #     Min = hitting_time_v1
    # if(Max>0):
    #     distance = (float(Max)/Min)-1
    if(method>1 and distance !=3):
        distance = distance*method
    return distance
                        

def calc_distances_ripple(vertices,list_vertices,rippleLists,part,max_layer,method):
    
    distances = {}
    
    cont = 0
    for v1 in vertices:
        list_v1 = rippleLists[v1]

        for v2 in list_vertices[cont]:
            list_v2 = rippleLists[v2]
            
            max_layer = min(len(list_v1),len(list_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                distance = ripple_distance(list_v1,list_v2,layer,max_layer,method)
                distances[v1,v2][layer] = distance
                
                
        cont += 1
    saveVariableOnDisk(distances,'distances-'+str(part))
    return


def exec_bfs(G,weight,workers,calcUntilLayer,mDgree,ripple):
    futures = {}
    rippleList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices,parts)
    max_depth = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getRippleListsVertices,G,weight,c,calcUntilLayer,mDgree,ripple)
            futures[job] = part
            part+= 1

        for job in as_completed(futures):
            ripple,depth = job.result()
            if(depth>max_depth):
                max_depth = depth
            v = futures[job]
            rippleList.update(ripple)

    logging.info("Saving rippleList on disk...")
    saveVariableOnDisk(rippleList,'RippleList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))


    return max_depth


def generate_distances_network_part1(workers):
    parts = workers
    weights_distances = {}
    for part in range(1,parts + 1):    
        
        logging.info('Executing part {}...'.format(part))
        distances = restoreVariableFromDisk('distances-'+str(part))
        
        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance

        logging.info('Part {} executed.'.format(part))

    for layer,values in weights_distances.items():
        saveVariableOnDisk(values,'weights_distances-layer-'+str(layer))
    return

def generate_distances_network_part2(workers):
    parts = workers
    graphs = {}
    for part in range(1,parts + 1):

        logging.info('Executing part {}...'.format(part))
        distances = restoreVariableFromDisk('distances-'+str(part))
        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs):
                    graphs[layer] = {}
                if(vx not in graphs[layer]):
                   graphs[layer][vx] = [] 
                if(vy not in graphs[layer]):
                   graphs[layer][vy] = [] 
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
        logging.info('Part {} executed.'.format(part))

    for layer,values in graphs.items():
        saveVariableOnDisk(values,'graphs-layer-'+str(layer))

    return

def generate_distances_network_part3():

    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        weights_distances = restoreVariableFromDisk('weights_distances-layer-'+str(layer))

        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}
    
        for v,neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                if (v,n) in weights_distances:
                    wd = weights_distances[v,n]
                else:
                    wd = weights_distances[n,v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        saveVariableOnDisk(weights,'distances_nets_weights-layer-'+str(layer))
        saveVariableOnDisk(alias_method_j,'alias_method_j-layer-'+str(layer))
        saveVariableOnDisk(alias_method_q,'alias_method_q-layer-'+str(layer))
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info('Weights created.')

    return


def generate_distances_network_part4():
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1


    logging.info("Saving distancesNets on disk...")
    saveVariableOnDisk(graphs_c,'distances_nets_graphs')
    logging.info('Graphs consolidated.')
    return

def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while(isPickle('alias_method_j-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_j = restoreVariableFromDisk('alias_method_j-layer-'+str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    saveVariableOnDisk(alias_method_j_c,'nets_weights_alias_method_j')

    return

def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while(isPickle('alias_method_q-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_q = restoreVariableFromDisk('alias_method_q-layer-'+str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    saveVariableOnDisk(alias_method_q_c,'nets_weights_alias_method_q')

    return

def generate_distances_network(workers):
    t0 = time()
    logging.info('Creating distance network...')

    os.system("rm "+returnPathripple2vec()+"/../pickles/weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm "+returnPathripple2vec()+"/../pickles/graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm "+returnPathripple2vec()+"/../pickles/distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathripple2vec()+"/../pickles/alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathripple2vec()+"/../pickles/alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 6: {}s'.format(t))
 
    return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q
