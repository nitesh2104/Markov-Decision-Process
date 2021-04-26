import logging
import numpy as np
import environments
import experiments
import multiprocessing
from datetime import datetime
from experiments import plotting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_detals, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_detals:
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        exp = experiment(details, verbose=verbose)
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    logger.info("Creating MDPs")
    logger.info("----------")

    envs = [
        # Grid World Problems - Frozen Lake
        {
            'env': environments.get_rewarding_frozen_lake_environment(),
            'name': 'frozen_lake',
            'readable_name': 'Frozen Lake (8x8)',
        },
        {
            'env': environments.get_large_with_rewards_and_slippery(),
            'name': 'large_frozen_lake',
            'readable_name': 'Frozen Lake (20x20)',
        }
    ]

    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS, env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(env['env'], env['name'], env['readable_name'], threads=multiprocessing.cpu_count() - 1, seed=seed))

    logger.info("----------")
    logger.info("Running experiments")

    timings = {}

    # Runs all the Algorithms (VI, PI and Q-Learning) sequentially
    run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', True, timings)
    run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', True, timings)
    run_experiment(experiment_details, experiments.QLearnerExperiment, 'Q', True, timings)

    logger.info(timings)
    logger.info("Plotting results")
    plotting.plot_results(envs)
