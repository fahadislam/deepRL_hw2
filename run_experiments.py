import os
import subprocess

space = 'SpaceInvadersDeterministic-v3'
breakout = 'BreakoutDeterministic-v3'

types = ['normal', 'double']
envs = [breakout, space]
netsizes = ['small', 'large']

mkdir_form = 'mkdir -p cache/%s-%s-%s'
log_form = 'cache/%s-%s-%s/run.log'


def run_form(gpu_id, type, env, netsize):
    gpu_id = str(gpu_id)
    return ['python', 'dqn_atari.py', '--gpu', gpu_id, '--type', type, '--env', env,
            '--netsize', netsize]


gpu_id = 0
processes = []
for t in types:
    for e in envs:
        for ns in netsizes:
            mkdir_cmd = mkdir_form % (e, t, ns)
            os.system(mkdir_cmd)
            run_cmd = run_form(gpu_id, t, e, ns)
            log_file = open(log_form % (e, t, ns), 'w')
            processes.append(subprocess.Popen(run_cmd, stdout=log_file, stderr=log_file))
            gpu_id = (gpu_id + 1) % 4
            
for process in processes:
    process.wait()
