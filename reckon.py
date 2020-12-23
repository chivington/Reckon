# imports
from utils.experiments import *
from utils.io import *
from models.fc import *


# numpy configuration
rng = np.random.default_rng(4)
np.set_printoptions(precision=5)


# Test Model
def main(args):
    # clear screen & display application info
    greet()

    # import SELEX data
    files = [
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R2_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R7_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R9_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R11_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R3_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R6_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R9_top_seq.csv'
    ]
    seqs, labels = combine_seq_rounds(files, threshold)     # encode & combine sequencing data; assign labels per threshold
    m, n = seqs.shape

    # define hyperparameters for multiple experiments
    experiments = [
        [10000, 50000, 0.001, 0.001, 0.90, 7000, 25, 0.8, 'Genomic - Baseline'],
        [10000, 50000, 0.0005, 0.001, 0.90, 7000, 25, 0.8, 'Genomic - Low Learning Rate'],
        [10000, 50000, 0.005, 0.001, 0.90, 7000, 25, 0.8, 'Genomic - High Learning Rate']
    ]

    # run experiments
    i = 0
    while i < len(experiments):
        experiment_data = create_experiment(experiments[i], i)
        acc_ratio, acc_range = run_experiment(FC_Net, experiment_data, seqs, labels)
        i += 1

    # MAIN END


if __name__ == "__main__":
    main(sys.argv)
