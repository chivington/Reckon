import sys, time
from utils.data import *
from utils.io import *

def create_experiment(params, i):
    prnt, epochs, alpha, lamb, l1, title, subtitle = params
    return {
        "title": f'{title}',            # experiment title
        "subtitle": f'{subtitle}',      # experiment subtitle
        "version": f'{i}',              # experiment version
        "prnt": prnt,                   # how frequently to display cost, acc & error
        "hyperparams": {                # model hyperparameters
            "epochs": epochs,           # number of gradient descent iterations
            "alpha": alpha,             # learning rate
            "lamb": lamb,               # regularization rate
            "l1": l1                    # width of layer 1
        }
    }

def save_experiment(experiment, out_dir):
    title, subtitle, version = experiment["title"], experiment["subtitle"], experiment["version"],
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print(f'\n Writing experiment data to disk - {out_dir}/results.txt')
    f = open(f'{out_dir}/results.txt', 'w+')
    f.write(f'# --- Reckon Experimental Analytics Report --- #\n  - experiment: {title} - v{version} {subtitle}\n  - duration: {experiment["duration"]}')
    f.write(f'\n\n# --- EXPERIMENT HYPERPARAMETERS --- #')
    [f.write(f'\n{param[1]}: {experiment["hyperparams"][param[1]]}') for param in enumerate(experiment["hyperparams"])]
    f.write(f'\n\n# --- LEARNED MODEL PARAMETERS --- #')
    [f.write(f'\n{layer[1]}: \n{experiment["learned_params"][layer[1]]}') for layer in enumerate(experiment["learned_params"])]
    f.write(f'\n\n# --- EXPERIMENT COST, ACC. & ERROR HISTORY --- #')
    [f.write(f'\n\n{history[1]}: \n{experiment["history"][history[1]]}') for history in enumerate(experiment["history"])]
    f.close()

def run_experiment(Model, experiment, X_train, Y_train, X_test, Y_test):
    epochs, alpha, lamb, l1 = experiment["hyperparams"].values()
    m, n = X_train.shape[0], X_train.shape[1]
    title, subtitle, version, prnt = experiment["title"], experiment["subtitle"], experiment["version"], experiment["prnt"]
    print(f'\n Running experiment "{title} - v{version} {subtitle}"\n   - {experiment["hyperparams"]}')

    # train & time model
    t1 = time.time()
    model = Model(alpha, lamb, epochs, [m, n, l1])
    hist = model.train(X_train, Y_train, m, n, prnt)
    t2 = time.time()
    duration = t2-t1

    # test model
    Z1, A1, Z2, A2 = model.compute_activations(X_test)
    cost, acc, err = model.test(Y_test, A2, b)
    print(f"\n Final Evaluation of Testing Performance\n   duration: ~{np.round(duration, 3)}s  cost: {np.round(cost, 4)}   acc: {np.round(acc, 4)}%\n")

    # plot cost and accuracy
    out_dir = f'./out/{title.replace(" ", "-")}/v{version}-{subtitle.replace(" ", "-")}'
    plot_data(duration, hist[0], 'Cost During Training', 'Epochs', 'Cost', False, out_dir)
    plot_data(duration, hist[1], 'Accuracy During Training', 'Epochs', 'Accuracy', False, out_dir)
    plot_data(duration, hist[2], 'Error During Training', 'Epochs', 'Error', False, out_dir)

    # write plots & learned params to disk
    experiment["learned_params"] = model.params
    experiment["duration"] = duration
    experiment["history"] = { "cost": hist[0][0], "accuracy": hist[1][0], "error": hist[2][0] }
    save_experiment(experiment, out_dir)
    return np.average(hist[2]) / np.max(hist[2]), np.max(hist[2]) - np.max(np.min(hist[2]), 0)
