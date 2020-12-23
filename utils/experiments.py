import sys
from utils.data import *
from utils.io import *

def create_experiment(params, i):
    prnt, epochs, alpha, lamb, train_size, threshold, l1, b, name = params
    return {
        "name": f'{name}',              # experiment name
        "version": f'{i}',              # experiment version
        "prnt": prnt,                   # how frequently to display cost, acc & error
        "hyperparams": {
            "epochs": epochs,           # number of gradient descent iterations
            "alpha": alpha,             # learning rate
            "lamb": lamb,               # regularization rate
            "train_size": train_size,   # dataset train/test ratio
            "threshold": threshold,     # sequencing read-count threshold for binding affinity classification
            "l1": l1,                   # width of layer 1
            "b": b                      # boundary for correct classification
        }
    }

def save_experiment(experiment):
    name, version = experiment["name"], experiment["version"],
    out_path = f'./out/v{version}-{name.replace(" ", "-")}'
    if not os.path.exists(out_path): os.makedirs(out_path)
    print(f'\n Writing experiment data to disk - {out_path}/results.txt')
    f = open(f'{out_path}/results.txt', 'w+')
    f.write(f'# --- Reckon Experimental Analytics Report - {name} ({version}) --- #')
    f.write(f'\n\n# --- EXPERIMENT HYPERPARAMETERS --- #')
    [f.write(f'\n{param[1]}: {experiment["hyperparams"][param[1]]}') for param in enumerate(experiment["hyperparams"])]
    f.write(f'\n\n# --- LEARNED MODEL PARAMETERS --- #')
    [f.write(f'\n\n{layer[1]}: \n{experiment["learned_params"][layer[1]]}') for layer in enumerate(experiment["learned_params"])]
    f.write(f'\n\n# --- EXPERIMENT COST, ACC. & ERROR HISTORY --- #')
    [f.write(f'\n\n{history[1]}: \n{experiment["history"][history[1]]}') for history in enumerate(experiment["history"])]
    f.close()

def run_experiment(Model, experiment, X, Y):
    epochs, alpha, lamb, train_size, threshold, l1, b = experiment["hyperparams"].values()
    name, prnt = experiment["name"], experiment["prnt"]
    print(f'\n Running experiment "{name}" - {experiment["hyperparams"]}')

    # shuffle data
    print("\n Creating training & test sets...")
    shuff = np.arange(m)
    np.random.shuffle(shuff)
    X_shuff, Y_shuff = X[shuff], Y[shuff]

    # split train & test sets
    X_train, Y_train = X_shuff[:int(train_size*X_shuff.shape[0])], Y_shuff[:int(train_size*X_shuff.shape[0])]
    X_test, Y_test = X_shuff[int(train_size*X_shuff.shape[0]):], Y_shuff[int(train_size*X_shuff.shape[0]):]

    # normalize data - numerically-encoded sequence normalization
    X_train_norm, X_test_norm = X_train/4, X_test/4

    # data dims
    m_train, n_train = X_train.shape[0], X_train.shape[1]
    m_test, n_test = X_test.shape[0], X_test.shape[1]

    # train models
    model = Model(alpha, lamb, epochs, [m_train, n_train, l1])
    hist = model.train(X_train, Y_train, m_train, n_train, b, prnt)

    # test model
    Z1, A1, Z2, A2 = model.compute_activations(X_test)
    cost, acc, err = model.test(Y_test, A2, b)
    print(f"\n Final Evaluation of Testing Performance\n cost: {np.round(cost, 5)}  acc: {np.round(acc, 5)}%\n")

    # plot cost and accuracy
    plot_data(hist[0], 'Cost During Training', 'Epochs', 'Cost', experiment["name"], experiment["version"])
    plot_data(hist[1], 'Accuracy During Training', 'Epochs', 'Accuracy', experiment["name"], experiment["version"])
    plot_data(hist[2], 'Error During Training', 'Epochs', 'Error', experiment["name"], experiment["version"])

    # write plots & learned params to disk
    experiment["learned_params"] = model.params
    experiment["history"] = { "cost": hist[0], "accuracy": hist[1], "error": hist[2] }
    save_experiment(experiment)
    return np.average(hist[2]) / np.max(hist[2]), np.max(hist[2]) - np.max(np.min(hist[2]), 0)
