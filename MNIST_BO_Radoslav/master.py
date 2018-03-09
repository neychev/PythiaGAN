import multiprocessing
import subprocess

def work(cmd):
    return subprocess.call(cmd, shell=False)

if __name__ == '__main__':
    suffix = '50 10 5 64'
    alpha_grid = [str(24*i) for i in range(15)]
    parameters = [' '.join((str(i%2+1), suffix, alpha_grid[i])) for i in range(15)]
    pool = multiprocessing.Pool(processes=15)
    pool.map(work, [' '.join(('python start_optimization.py', par)).split(' ') for par in parameters])
