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
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    avg_reward.columns = ["0", "timesteps", "avg reward"]
    avg_reward = avg_reward[["timesteps", "avg reward"]]
    ax[0, 0].plot(avg_reward['timesteps'], avg_reward['avg reward'])
    ax[0, 0].set_title('average reward')

    est_q = pd.read_csv(args.cql_estimated_q_path, header=None)
    est_q.columns = ["0", "timesteps", "estimated q"]
    est_q = est_q[["timesteps", "estimated q"]]
    ax[0, 1].plot(est_q['timesteps'], est_q['estimated q'])
    ax[0, 1].set_title('estimated q values')

    true_q = pd.read_csv(args.cql_true_q_path, header=None)
    true_q.columns = ["0", "timesteps", "true q"]
    true_q = true_q[["timesteps", "true q"]]
    ax[1, 0].plot(true_q['timesteps'], true_q['true q'])
    ax[1, 0].set_title('true q values')

    fqe_estimated = pd.read_csv(args.fqe_estimated_q_path, header=None)
    fqe_estimated.columns = ["0", "timesteps", "estimated q(fqe)"]
    fqe_estimated = fqe_estimated[["timesteps", "estimated q(fqe)"]]

    fqe_true = pd.read_csv(args.fqe_true_q_path, header=None)
    fqe_true.columns = ["0", "timesteps", "true q(fqe)"]
    fqe_true = fqe_true[["timesteps", "true q(fqe)"]]

    ax[1, 1].plot(fqe_estimated['timesteps'], fqe_estimated['estimated q(fqe)'])
    ax[1, 1].plot(fqe_true['timesteps'], fqe_true['true q(fqe)'])
    ax[1, 1].set_title('fqe true q vs estimated q values')
    plt.savefig('plot.png')  # save fig
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
