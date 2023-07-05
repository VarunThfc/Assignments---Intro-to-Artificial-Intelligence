import csv
import numpy as np
import math
from scipy.spatial import distance_matrix
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import time


def load_data(filepath):
    with open(filepath, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        data_list = list()
        for di in reader:
            data_list.append(di)
    return data_list;
     
     
def calc_features(row):
    row_list = list()
    attack = 0;
    spAttack = 0;
    speed = 0;
    defense = 0;
    spDefense = 0;
    hp = 0;
    attack,  spAttack, speed, defense, spDefense, hp = int(row['Attack']), int(row['Sp. Atk']), int(row['Speed']), int(row['Defense']), int(row['Sp. Def']), int(row['HP'])
    row_list = [attack, spAttack, speed, defense, spDefense, hp];
    num_array = np.array(row_list)
    
    return num_array;

def hac(features):
    n = len(features)
    distance_matrix = np.ones((2*n - 1, 2*n - 1)) * np.inf
    features = np.array(features)
    comp1 = np.sum(features * features, axis = 1).reshape(-1,1)
    comp2 =  np.sum(features * features, axis = 1)

    comp3 = 2 * features @ features.T

    distance_matrix[:n,:n] = np.sqrt(comp1 + comp2 - comp3)
   
    clusters_info = {i: [[features[i]], 1] for i in range(n)}
    
    Z = np.zeros((n-1, 4))
    for k in range(n-1):
        min_distance = np.inf
        i = 0; j = 0;
        for c1 in clusters_info:
            for c2 in clusters_info:
                if c1 < c2 and distance_matrix[c1,c2] <= min_distance:
                    if distance_matrix[c1, c2] == min_distance:
                        if c1 > i or (c1 == i and c2 >= j):
                            continue
                    i = c1; j = c2
                    min_distance = distance_matrix[i, j]
                    
        clusters_info[n + k] = [clusters_info[i][0] + clusters_info[j][0], clusters_info[i][1] + clusters_info[j][1]]
        Z[k,0] = i
        Z[k,1] = j
        Z[k,2] = distance_matrix[i,j]
        Z[k,3] = clusters_info[n+k][1]
        for index in clusters_info:
            distance_matrix[index, n+k] = max(distance_matrix[i,index], distance_matrix[j,index], distance_matrix[index,i], distance_matrix[index,j])
            distance_matrix[n+k, index] = max(distance_matrix[i,index], distance_matrix[j,index], distance_matrix[index,i], distance_matrix[index,j])
        clusters_info.pop(i, None); clusters_info.pop(j, None)
    return Z
        
def mshow_hac(Z, names):
    fig, (ax1) = plt.subplots(1)
    hierarchy.dendrogram(Z, labels=names, ax = ax1, leaf_rotation=90)
    ax1.set_title(label = "N = " + str(len(names)))
    fig.tight_layout()
    plt.show()
