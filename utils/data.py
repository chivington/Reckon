import csv
import numpy as np

np.set_printoptions(precision=5)

def import_sequence_data(seq_data, threshold):
    print(f"  Importing sequence data - {seq_data}")
    path = './data/genomic'
    with open(path + seq_data, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))[1:-1]
        m, n = len(data), len(data[0][0].split(",")[1])
        X, Y = np.zeros([m,n]), np.zeros([m,1])
        seq_keys = {}
        for i in range(m):
            row = data[i][0].split(",")
            seq = row[1]
            seq = seq.replace("A", "1")
            seq = seq.replace("T", "2")
            seq = seq.replace("C", "3")
            seq = seq.replace("G", "4")
            for j in range(n):
                X[i,j] = float(seq[j])
            Y[i,0] = 0 if (int(row[2]) < threshold) else 1
            seq_keys[seq] = 0 if (int(row[2]) < threshold) else 1
        return X, Y, m, n, seq_keys

def combine_seq_rounds(files, threshold):
    print(f'\n Combining sequencing data according to threshold: {threshold}')
    master_keys = {}
    for i, f in enumerate(files):
        X, Y, m, n, seq_keys = import_sequence_data(f, threshold)
        if (i == len(files)):
            for j, key in enumerate(master_keys):
                if (seq_keys[key]): master_keys[key] = Y[j]
                else: master_keys[key] = 0
        for j, key in enumerate(seq_keys):
            master_keys[key] = Y[j]
        seqs = np.matrix([list(seq) for seq in list(master_keys.keys())], dtype=np.float64)
        labels = np.matrix(list(master_keys.values()), dtype=np.float64)
    print(f"\n Combining sequence data...")
    return seqs, labels

def one_hot_encode(X):
    one_hot = np.zeros([X.shape[0], X.shape[1], 4])
    for i in range(X.shape[0]):
        for j in range(len(X[i])):
            base = X[i,j]
            one_hot[i,j,base-1] = 1
    return one_hot
