import gensim
import sys
import numpy as np
import random
from sklearn import linear_model
import operator
import matplotlib.pyplot as plt


from gram_schmidt import gramschmidt

# Running sparse coding using majorization

base_path = "/Users/u6042446/Desktop/ali_files/sparse_coding/data/glove/"
PATH_TO_SPARSE_CODES = base_path+"sparse_vectors_200_0.001_50.txt"

# List of parameters
dim_atoms = 200

if __name__ == "__main__":

    atom_map = {}
    with open(PATH_TO_SPARSE_CODES,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            word = data[0]
            codes = data[1:]
            for item in codes:
                split = item.split(":")
                atom_index = int(split[0])
                weight = abs(float(split[1]))
                if not atom_index in atom_map: atom_map[atom_index] = {}
                atom_map[atom_index][word] = weight


    sorted_atom_map = {}
    atom_app = {}
    for atom_index in atom_map:
        s = sorted(atom_map[atom_index].items(),key=operator.itemgetter(1),reverse=True)
        sorted_atom_map[atom_index] = s
        atom_app[atom_index] = len(s)

    sorted_atom_app = sorted(atom_app.items(),key=operator.itemgetter(1),reverse=True)

    v = []
    [v.append(x[1]/400000) for x in sorted_atom_app]
    sys.exit(0)
    plt.plot(range(0,200),v)
    plt.show()







