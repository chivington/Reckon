import os, sys
import matplotlib.pyplot as plt

def plot_data(data, title, x_label, y_label, name, version):
    fig, ax1 = plt.subplots(figsize=[10,6])
    plt.title(f'{name} (v{version}) {title}', fontsize=16, fontweight='bold')
    plt.style.use('seaborn-whitegrid')
    ax1.tick_params(axis='y')
    ax1.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax1.plot(data, '-', color='#0af', label=y_label)
    fig.legend(frameon=True, borderpad=1, facecolor='#fff', framealpha=1, edgecolor='#777', shadow=True)
    if not os.path.exists(f'./out/v{version}-{name.replace(" ", "-")}'): os.makedirs(f'./out/v{version}-{name.replace(" ", "-")}')
    plt.savefig(f'./out/v{version}-{name.replace(" ", "-")}/{title.replace(" ", "-")}-plot.png')
    plt.close("all")

def underline(m=''):
    if m == '': print('\n No message passed.'); return
    else: print(m);[sys.stdout.write((' ' if m[i]==' ' else '-')if i==0 else('\n'if i==len(m)+1 else '-'))for i in range(len(m)+1)]

def clear(m=''):
    os.system('clear'if os.name=='posix'else'cls'); print(m)

def greet():
    clear('')
    underline(f' Welcome to Reckon')
