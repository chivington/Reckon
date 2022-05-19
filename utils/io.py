import os, sys
import matplotlib.pyplot as plt

def plot_data(duration, data, plt_title, x_label, y_label, display, out_dir):
    fig, ax1 = plt.subplots(figsize=[10,6])
    plt.title(f'{plt_title} (~{duration}s)', fontsize=16, fontweight='bold')
    plt.style.use('seaborn-whitegrid')
    ax1.tick_params(axis='y')
    ax1.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax1.plot(data, '-', color='#0af', label=y_label)
    fig.legend(frameon=True, borderpad=1, facecolor='#fff', framealpha=1, edgecolor='#777', shadow=True)
    if display: plt.show()
    else:
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        plt.savefig(f'{out_dir}/{plt_title.replace(" ", "-").lower()}-plot.png')
        plt.close("all")

def underline(m=''):
    if m == '': print('\n No message passed.'); return
    print(m)
    for i in range(len(m)):
        if i==0: sys.stdout.write(' ' if m[i]==' ' else '-')
        else: sys.stdout.write('-\n'if i==len(m)-1 else '-')

def clear(m=''):
    os.system('clear' if os.name=='posix' else 'cls'); print(m)

def greet():
    clear('')
    underline(f' Welcome to Reckon')
