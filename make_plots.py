import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
import pandas as pd

matplotlib.rcParams['font.size'] = 12

def assign_color(path, params):
    assignments = {'joint':'tab:gray', 'disagreement':'tab:green',
                   'concat':'tab:purple', 'visual':'#ff7f0e', 'fft':'#1f77b4'}
    for method in assignments.keys():
        if method in path or method in params:
            if 'noise' in params:
                alpha = 0.5
            else:
                alpha = 1.0
            return assignments[method], alpha
    return None, 1.0

def assign_label(path, params):
    if 'b-' in path:
        label = 'Visual prediction curiosity'
    elif 'concat' in path:
        label = 'Audio-visual prediction'
    else:
        label = 'Audio-visual association (ours)'
    if 'noise' in params:
        if '(ours)' in label:
            label = label[:-7]
        label += ' ' + params.split('noise')[1] + 'with noise'
    if 'joint' in path:
        label = 'Combined curiosities (ours)'
    if 'normalize' in path:
        label += 'losses normalized'
    if 'disagreement' in params:
        label = 'Disagreement'
    return label

def plot_run(paths, x='tcount', ys='eprew', label='', assign_colors=True, params=''):
    for y in ys.split(','):
        all_runs = []
        nframes = None
        for path in paths:
            f = open(path + '/progress.csv', 'r')
            if 'concat' in path and not args.concat:
                continue

            try:
                df = pd.read_csv(f)
                all_runs.append(df[y])
                nframes = df[x]
            except:
                print("Exception while reading file", sys.exc_info()[0], path)
                continue

        if len(all_runs) == 0:
            return

        color = None
        alpha = 0.75
        if assign_colors:
            color, alpha = assign_color(path, params)
        if label != '':
            if params != '':
                label = assign_label(path, params)
            if len(ys.split(',')) > 1:
                label += ' ' + y
        min_length = min([len(run) for run in all_runs])
        all_runs = np.asarray([run[:min_length] for run in all_runs])
        mean_run = np.mean(all_runs, axis=0)
        nframes = nframes[:min_length] * 4

        ax.plot(nframes, mean_run, '-', label=label, color=color, alpha=alpha)
        if all_runs.shape[0] > 1:
            error = np.std(all_runs, axis=0)
            alpha = 0.2
            if 'noise' in params:
                alpha = 0.1
            ax.fill_between(nframes, mean_run-error, mean_run+error,
                            alpha=alpha, linewidth=0.0, color=color)

def get_paths(paths_str):
    paths = []
    for path in paths_str.split(','):
        if '*' in path:
            try:
                paths.extend(os.popen('ls -d $TMPDIR/0*' + path[1:] + '*/ 2> /dev/null').read().split())
            except:
                print("Exception: paths not found")
                continue
        else:
            paths.append(path)
    return paths

def create_params(path):
    if ('_s-' in path) != args.sticky:
        return None

    if 'noise' in path and not args.noise:
        return None

    if 'joint' in path:
        return None

    if args.mean:
        params = '_'.join(path.split('_')[-6:])
        if 'noise' in path:
            noise_amt = float(path.split('noise')[1].split('-')[0])
            if noise_amt > 0.1:
                return None
            if noise_amt != 0:
                params = params[:-1]
                params += 'noise'
        if 'concat' in path:
            concat_type_real = ' ' + path.split('concat')[0].split('breakout')[-1][1:-1]
            concat_type = path.split('concat')[1].split('_')[0][:-1]
            if len(concat_type) == 0 or concat_type == '-both':
                params += '-concat'
            else:
                params += concat_type
        if '_s-' in path:
            params += '_sticky'
        if 'disagreement' in path or 'openai' in path:
            params = get_disagreement_params(path)
        if 'joint' in path:
            params += '_joint'
        if 'normalized' in path:
            params += '-normalized'
        if 'unweighted' in path:
            params += '-unweighted'
    return params


def create_figure():
    if args.all:
        if args.noise:
            envs = ['Asterix', 'MsPacman', 'SpaceInvaders']
            fig, axs = plt.subplots(1, 3, sharex=True, figsize=(18, 6))
        else:
            envs = ['AirRaid', 'Alien', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', \
                    'BeamRider', 'Breakout', 'MsPacman', 'Qbert', 'Seaquest', 'SpaceInvaders']
            if args.sticky:
                fig, axs = plt.subplots(2, 6, sharex=True, figsize=(18, 6))
            else:
                fig, axs = plt.subplots(3, 4, sharex=True, figsize=(20, 15))
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        left=False, right=False, labelcolor='none')
        plt.grid(False)
    else:
        fig, ax = plt.subplots(1, 1)
        axs = np.array(ax)
        envs = ['']
    fsize = 32
    if args.noise:
        fsize = 20
    plt.xlabel("\nFrames (millions)\n", fontsize=fsize)
    plt.ylabel("Extrinsic Reward Per Episode\n", fontsize=fsize)
    return fig, axs, envs

def finish_plot():
    if args.xlim:
        plt.xlim((-5*1e6, args.xlim * 1e6))
    if args.ylim:
        plt.ylim((-0.0 * args.ylim, args.ylim))

    mean_str = '_mean' if args.mean else ''

    if args.all:
        sticky_str = '_sticky' if args.sticky else ''
        noise_str = '_noise' if args.noise else ''
        save_name = 'all' + sticky_str + mean_str + noise_str + "_" + args.y + '.png'
    else:
        save_name = path.split('/')[-1] + '_' + str(len(ps)) + \
                        '_' + str(args.y) + mean_str + '.png'
    plt.savefig(save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str)
    parser.add_argument('-x', type=str, default='tcount')
    parser.add_argument('-xlim', type=int, default=200)
    parser.add_argument('-ylim', type=float)
    parser.add_argument('-y', type=str, default='eprew')
    parser.add_argument('--mean', type=bool, default=False)
    parser.add_argument('--all', type=bool, default=False)
    parser.add_argument('--sticky', type=bool, default=False)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--concat', type=bool, default=False)
    parser.add_argument('--assign_colors', type=bool, default=False)

    args = parser.parse_args()
    fig, axs, envs = create_figure()
    env_index = 0

    for ax in axs.flat:
        runs_by_params = defaultdict(list)
        if args.all:
            if env_index >= len(envs):
                break
            ps = get_paths('*' + envs[env_index])
        else:
            ps = get_paths(args.paths)
        if len(ps) == 0:
            print("No paths found")
            sys.exit()

        for path in ps:
            # Remove trailing slash from pathname
            if path[-1] == '/':
                path = path[:-1]

            if args.mean:
                params = create_params(path)
                if params is not None:
                    runs_by_params[params].append(path)
            else:
                run_name = path.split('/')[-1]
                plot_run([path], args.x, args.y, label=run_name, assign_colors=args.assign_colors)

        if args.mean:
            for params, runs in runs_by_params.items():
                # Only add labels for legend in last plot
                label = params
                if args.all and env_index != len(envs) - 1:
                    label = ''
                plot_run(runs, args.x, args.y, label=label, assign_colors=True, params=params)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
        if not args.all or env_index == len(envs) - 1:
            ncol = 3
            fsize = 25
            loc = (0.05, 0.01)
            if args.noise:
                ncol = 2
                fsize = 18
                loc = (0.25, 0.01)
            leg = fig.legend(prop={'size': fsize}, ncol=ncol, loc=loc)

            for line in leg.get_lines():
                line.set_linewidth(4.0)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(envs[env_index], fontsize=16)
        ax.set_xlim((-5 * 1e6, args.xlim * 1e6))
        env_index += 1

    fig.tight_layout(pad=0.5)
    finish_plot()
