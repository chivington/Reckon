import csv
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=5)
rng = np.random.default_rng(4)


# data-loading functions
def import_SELEX_data(threshold, train_ratio):
    print(f"\n Importing SELEX data:")
    files = [
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R2_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R7_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R9_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R11_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R3_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R6_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R9_top_seq.csv'
    ]
    master_keys = {}
    for i,f in enumerate(files):
        print(f"   - {f}")
        with open('./data/genomic' + f, newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))[1:-1]
            m, n = len(data), len(data[0][0].split(",")[1])
            X, Y = np.zeros([m,n]), np.zeros([m,1])
            seq_keys = {}
            for i in range(m):
                row = data[i][0].split(",")
                seq = row[1]
                for j,b in enumerate(["A", "T", "C", "G"]): seq = seq.replace(b, f"{j+1}")
                for k in range(n): X[i,k] = float(seq[k])
                Y[i,0] = 0 if (int(row[2]) < threshold) else 1
                seq_keys[seq] = 0 if (int(row[2]) < threshold) else 1
        if (i == len(files)):
            for j, key in enumerate(master_keys): master_keys[key] = Y[j] if seq_keys[key] else 0
        for j, key in enumerate(seq_keys): master_keys[key] = Y[j]
        seqs = np.matrix([list(seq) for seq in list(master_keys.keys())], dtype=np.float64)
        labels = np.matrix(list(master_keys.values()), dtype=np.float64)
        shuff = np.arange(seqs.shape[0])
        np.random.shuffle(shuff)
        X_shuff, Y_shuff = seqs[shuff], labels[shuff]
        idx = int(np.floor(train_ratio * seqs.shape[0]))
        X_train, Y_train, X_test, Y_test = seqs[:idx], labels[:idx], seqs[idx:], labels[idx:]
    print(f'\n Combining sequencing data according to threshold: {threshold}')
    return X_train, Y_train, X_test, Y_test

def import_mnist_data():
    print(f"\n Importing MNIST data:")
    import mnist
    mnist_data = mnist.MNIST('./data/image/mnist')
    X_train, Y_train = map(np.array, mnist_data.load_training())
    X_test, Y_test = map(np.array, mnist_data.load_testing())
    return X_train/255.0, Y_train, X_test/255.0, Y_test

def display_example(X, Y, data_type):
    i = rng.integers(X.shape[0])
    example, label = X[i], Y[i]
    print(f"\n Displaying random example ({i}):")
    if data_type == 'sequence':
        print(f'\n seq: {X[i]}\n label: {Y[i]}')
    elif data_type == 'mnist':
        print("  Close to continue...")
        plt.figure(figsize=plt.figaspect(1.0))
        plt.subplot(1,1,1)
        plt.imshow(X[i].reshape([28,28]), cmap=plt.cm.gray)
        plt.title(f'Random MNIST Digit: {Y[i]}', fontsize=20)
        plt.show()


# encoding functions
# def one_hot_encode_dna(X):
#     one_hot = np.zeros([X.shape[0], X.shape[1], 4])
#     for i in range(X.shape[0]):
#         for j in range(len(X[i])):
#             base = X[i,j]
#             one_hot[i,j,base-1] = 1
#     return one_hot

def one_hot_encode_mnist(self,Y):
    n = Y.shape[0]
    one_hot = np.zeros((n,self.k))
    for i in range(n): one_hot[i,Y[i]] = 1
    return one_hot
