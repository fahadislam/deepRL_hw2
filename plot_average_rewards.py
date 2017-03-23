import os
import argparse
import subprocess
import numpy as np 

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Plot average rewards across episode')
parser.add_argument('--dir', help='model directory')
parser.add_argument('--epoch', default=15, type=int, help='maximum epoch')
args = parser.parse_args()

N = args.epoch
epochs = []
avg_lens = np.zeros(N)
std_lens = np.zeros(N)
avg_rews = np.zeros(N)
std_rews = np.zeros(N)

res_path = os.path.join(args.dir, 'evaluation.txt')
if os.path.exists(res_path):
    res_file = open(res_path, 'r')
    lines = res_file.readlines()
    res_file.close()
    for i in np.arange(len(lines)):
        line = lines[i].split()
        if int(line[0]) not in epochs:
            epochs.append(int(line[0]))
            avg_lens[i] = float(line[1])
            std_lens[i] = float(line[2])
            avg_rews[i] = float(line[3])
            std_rews[i] = float(line[4])

if len(epochs) != N:
    processes = []
    for i in range(0,args.epoch+1):
        if i not in epochs:
            process = subprocess.Popen(['python', 'dqn_atari.py', '--mode', 'test', '--run', '1', '--epoch', '%d'%i])
            processes.append(process)
    for process in processes:
        process.wait()

res_file = open(res_path, 'r')
lines = res_file.readlines()
res_file.close()
for i in np.arange(N):
    line = lines[i].split()
    epochs[i] = int(line[0])
    avg_lens[i] = float(line[1])
    std_lens[i] = float(line[2])
    avg_rews[i] = float(line[3])
    std_rews[i] = float(line[4])

epochs = np.array(epochs)
I = np.argsort(epochs)
epochs = epochs[I]
avg_lens = avg_lens[I]
std_lens = std_lens[I]
avg_rews = avg_rews[I]
std_rews = std_rews[I]

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
ax = axs[0]
ax.errorbar(epochs, avg_lens, yerr=std_lens, fmt='--o')
ax.set_title('Average Episode Length across Epochs')

ax = axs[1]
ax.errorbar(epochs, avg_rews, yerr=std_rews, fmt='--o')
ax.set_title('Average Reward per Episode across Epochs')

plt.savefig(os.path.join(args.dir, 'eval.pdf'))
plt.show()
