import argparse

import matplotlib.pyplot as plt
import pandas as pd


########################################################
# Author: Shyamal H Anadkat | AIPI530 | Fall 2021      #
########################################################

def main(args):
    # logging for debugging
    print("=========================")
    print("CQL True Q Logs Path:  ", args.cql_true_q_path)
    print("CQL Estimated Q Logs Path:  ", args.cql_estimated_q_path)
    print("CQL Avg Reward Logs Path:  ", args.cql_reward_path)
    print("FQE True Q Logs Path:  ", args.fqe_true_q_path)
    print("FQE Estimated Q Logs Path:  ", args.fqe_estimated_q_path)
    print("=========================")

    avg_reward = pd.read_csv(args.cql_reward_path, header=None)
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(avg_reward.iloc[:, 0], avg_reward.iloc[:, 2])
    ax[0, 0].set_title('average reward')

    est_q = pd.read_csv(args.cql_estimated_q_path, header=None)
    ax[0, 1].plot(est_q.iloc[:, 0], est_q.iloc[:, 2])
    ax[0, 1].set_title('estimated q values')

    true_q = pd.read_csv(args.cql_true_q_path, header=None)
    ax[1, 0].plot(true_q.iloc[:, 0], true_q.iloc[:, 2])
    ax[1, 0].set_title('true q values')

    fqe_estimated = pd.read_csv(args.fqe_estimated_q_path, header=None)
    fqe_true = pd.read_csv(args.fqe_true_q_path, header=None)

    ax[1, 1].plot(fqe_estimated.iloc[:, 0], fqe_estimated.iloc[:, 2])
    ax[1, 1].plot(fqe_true.iloc[:, 0], fqe_true.iloc[:, 2])
    ax[1, 1].set_title('fqe true q vs estimated q values')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cql_true_q_path',
                        type=str,
                        default='/content/offlinerl/d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/true_q_value.csv')
    parser.add_argument('--cql_estimated_q_path',
                        type=str,
                        default='/content/offlinerl/d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/init_value.csv')
    parser.add_argument('--cql_reward_path',
                        type=str,
                        default='/content/offlinerl/d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/environment.csv')
    parser.add_argument('--fqe_true_q_path',
                        type=str,
                        default='/content/offlinerl/d3rlpy_logs/FQE_hopper-bullet-mixed-v0_1/true_q_value.csv')
    parser.add_argument('--fqe_estimated_q_path',
                        type=str,
                        default='/content/offlinerl/d3rlpy_logs/FQE_hopper-bullet-mixed-v0_1/init_value.csv')
    args = parser.parse_args()
    main(args)
