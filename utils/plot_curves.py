import sys
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_curve_base(log_dir, color='black', start_iter=1, exp_name=''):
    iters = []
    evals = []
    with open(log_dir, 'r') as logf:
        line = logf.readline()
        while line:
            line = line.split(' ')
            for k in range(0, len(line), 2):
                if line[k] == 'iter':
                    iters.append(float(line[k + 1]))
                if line[k] == 'eval':
                    evals.append(float(line[k + 1]))
            line = logf.readline()
    for i in range(len(iters)):
        if iters[i] >= start_iter:
            iters = iters[i:]
            evals = evals[i:]
            break
    plt.plot(iters, evals, '-', color=color, linewidth=1, label=exp_name)
    return min(evals), max(evals)

def plot_curve(log_dir, color='black', start_iter=1000):
    fig = plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots()
    min_eval, max_eval = plot_curve_base(log_dir, color, start_iter)
    fig_dir = log_dir.split('.')[0] + '.pdf'
    #plt.yscale('log')
    plt.yticks([t * 0.02 for t in
        range(int(min_eval/0.02), int(max_eval/0.02), 1)])
    fig.axes[0].grid()
    fig.savefig(fig_dir)

def plot_curves(log_dirs, start_iter=1000):
    fig = plt.figure(figsize=(10, 10))
    min_evals = []
    max_evals = []
    rainbow = cm.get_cmap('Set2')
    for i, log_dir in enumerate(log_dirs):
        exp_name = log_dir.split('/')[-2]
        min_eval, max_eval = plot_curve_base(log_dir, exp_name=exp_name, color=rainbow(i / len(log_dirs)))
        min_evals.append(min_eval)
        max_evals.append(max_eval)

    min_eval = min(min_evals)
    max_eval = max(max_evals)
    #plt.yscale('log')
    plt.yticks([t * 0.02 for t in
        range(int(min_eval/0.02), int(max_eval/0.02)+2, 1)])
    fig.axes[0].grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #print(sys.argv)
    if len(sys.argv) <=1:
        print(".log files needed")
        exit(0)

    # check if args are legal
    for arg in sys.argv[1:]:
        if arg.split('.')[-1] != 'log':
            print(arg, 'is not a .log file')
            exit(0)

    plot_curves(sys.argv[1:])
