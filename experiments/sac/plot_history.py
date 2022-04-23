from src.libs.plot_utils import load_history, plot_moving_average_reward, plot_metric
import argparse
import yaml
from pathlib import Path


def history_parser():
    parser = argparse.ArgumentParser(description='SAC')
    parser.add_argument('--history_dir', '-hd', type=str, default='data/experiment_data/sweep_exp/',
                        required=False,
                        help='path to save logs of learning')
    parser.add_argument('--plots_dir', '-pd', type=str, default='data/experiment_data/sweep_plots/',
                        required=False,
                        help='Path to plots output directory')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = history_parser()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)

    additional_params = params_all['sac_params'].get('additional_params')
    general_params = params_all['sac_params'].get('general_params')
    neural_network_params = params_all['sac_params'].get('neural_network_params')
    experiment_params = params_all['sac_params'].get('experiment_params')
    param_sweep = params_all['param_sweep']

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)

    for filename, df in load_history(data_path=Path(args.history_dir)):
        plot_moving_average_reward(dataframe=df, filename=filename.name, output_dir=plots_dir, title='Moving average',
                                   xlabel='Episode', ylabel='Moving average reward')
        plot_metric(dataframe=df, filename=filename.name, output_dir=plots_dir, title='Metric',
                                   xlabel='Episode', ylabel='Metric')
