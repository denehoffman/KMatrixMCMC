#!/usr/bin/env python3

import numpy as np
import subprocess
import matplotlib.pyplot as plt

def main():
    ms = np.linspace(0.5, 2.0, 500)
    #vals = []
    #for m in track(ms):
    #    vals.append(float(subprocess.run(['./run_mcmc', str(m)], capture_output=True, text=True).stdout))
    #print(vals)
    vals = [float(v) for v in subprocess.run(['./run_mcmc', *[str(m**2) for m in ms]],
                                             capture_output=True,
                                             text=True).stdout.strip().split('\n')]
    print(vals)
    plt.plot(ms, vals)
    plt.ylim(0.0, 200)
    plt.show()

if __name__ == "__main__":
    main()
