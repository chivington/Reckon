# imports
from utils.experiments import *
from utils.io import *
from models.selex import SELEX_Net
from models.mnist import MNIST_Net


# numpy configuration
np.set_printoptions(precision=5)


# test Model
def main(args):
    # clear screen & display application info
    greet()

    # import, encode & combine SELEX sequencing data; assign labels per threshold
    # X_train, Y_train, X_test, Y_test = import_SELEX_data(7000, 0.9)
    # display_example(X_train, Y_train, 'sequence')


    # import MNIST train & test datasets
    X_train, Y_train, X_test, Y_test = import_mnist_data()
    display_example(X_train, Y_train, 'mnist')

    # define hyperparameters for multiple experiments: print frequency, optimization iterations, learning rate,
    #   regularization weight, train/test ratio, hidden layer width, classification boundary
    experiments = [
        [2000, 10000, 0.001, 0.001, 25, 'MNIST Classification w/ Selex Model', 'Baseline'],
        # [2000, 10000, 0.001, 0.001, 25, 'SELEX Binding Affinity Classification', 'Baseline'],
        # [2000, 10000, 0.001, 0.001, 25, 'MNIST Digit Classification', 'Baseline'],
    ]

    # run experiments
    i = 0
    while i < len(experiments):
        experiment_params = create_experiment(experiments[i], i)
        acc_ratio, acc_range = run_experiment(SELEX_Net, experiment_params, X_train, Y_train, X_test, Y_test)
        i += 1

    # MAIN END


if __name__ == "__main__":
    main(sys.argv)
