import argparse

from sklearn.model_selection import train_test_split

import d3rlpy
from d3rlpy.algos import CQL
from d3rlpy.datasets import get_pybullet
from d3rlpy.gpu import Device
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.metrics.scorer import true_q_value_scorer
from d3rlpy.ope import FQE


########################################################
# Author: Shyamal H Anadkat | AIPI530 | Fall 2021      #
########################################################

def main(args):
    # returns pybullet dataset and environment (dataset being passed in args)
    dataset, env = get_pybullet(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    # set conservative weight
    if "medium-v0" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    device = None if args.gpu is None else Device(args.gpu)

    # logging for debugging
    print("=========================")
    print("Q FUNQ :  ", args.q_func)
    print("USE GPU : ", device)
    print("DATASET : ", args.dataset)
    print("EPOCHS (CQL) : ", args.epochs_cql)
    print("EPOCHS (FQE) : ", args.epochs_fqe)
    print("=========================")

    # Train CQL on given environment/dataset from pybullet (default is hopper-bullet-mixed-v0)

    cql = CQL(actor_learning_rate=1e-4,
              critic_learning_rate=3e-4,
              temp_learning_rate=1e-4,
              actor_encoder_factory=encoder,
              critic_encoder_factory=encoder,
              batch_size=256,
              n_action_samples=10,
              alpha_learning_rate=0.0,
              conservative_weight=conservative_weight,
              use_gpu=device)

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=args.epochs_cql,
            save_interval=10,
            scorers={
                # Returns scorer function of evaluation on environment (mean_episode_return)
                # Average reward vs training steps
                'environment': evaluate_on_environment(env),

                # Returns mean estimated action-values at the initial states
                # Estimated Q vs training steps
                'init_value': initial_state_value_estimation_scorer,

                # Returns true q values
                # True Q vs training steps
                "true_q_value": true_q_value_scorer
            },

            # log files are consistently named without timestamp (allows overwriting)
            with_timestamp=False,
            verbose=True,
            experiment_name=f"CQL_{args.dataset}_{args.seed}")

    # Train OPE (FQE) to evaluate the trained policy
    fqe = FQE(algo=cql,
              n_epochs=args.epochs_fqe,
              q_func_factory='qr',
              learning_rate=1e-4,
              use_gpu=device,
              encoder_params={'hidden_units': [1024, 1024, 1024, 1024]})

    fqe.fit(dataset.episodes,
            n_epochs=args.epochs_fqe,
            eval_episodes=dataset.episodes,
            scorers={
                # Returns mean estimated action-values at the initial states
                # Estimated Q vs training steps
                'init_value': initial_state_value_estimation_scorer,

                # Returns Soft Off-Policy Classification metrics
                'soft_opc': soft_opc_scorer(600),

                # Returns true q values
                "true_q_value": true_q_value_scorer
            },

            # log files are consistently named without timestamp (allows overwriting)
            with_timestamp=False,
            verbose=True,
            experiment_name=f"FQE_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='hopper-bullet-mixed-v0')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs_cql', type=int, default=10)  # epochs for cql being passed in as arg
    parser.add_argument('--epochs_fqe', type=int, default=10)  # epochs for fqe being passed in as arg
    parser.add_argument('--q-func',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    main(args)
