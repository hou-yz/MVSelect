import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    precs = np.loadtxt(
        '/home/houyz/Code/MVselect/logs/modelnet40_12/resnet18_max_down1_lr5e-05_b8_e10_dropcam0.0_2023-02-26_04-03-42/prec_94.0_Lstrategy85.8_Rstrategy85.5_theory86.0_avg69.1.txt')
    precs = precs[len(precs) // 2:]
    N = len(precs)

    plt.figure(figsize=(5, 5))
    plt.imshow(precs, cmap='Blues')
    plt.xticks(np.arange(N), np.arange(N) + 1)
    plt.xlabel('second view')
    plt.yticks(np.arange(N), np.arange(N) + 1)
    plt.ylabel('initial view')
    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = plt.text(j, i, precs[i, j], ha="center", va="center", )
    plt.tight_layout()
    plt.show()

    # only mvselect
    # prob = np.zeros([N, N])
    # prob[0, 10] = 1
    # prob[1, 10] = 1
    # prob[2, 10] = 1
    # prob[3, 10] = 1
    # prob[4, [1, 9]] = [0.88, 0.12]
    # prob[5, 10] = 1
    # prob[6, 10] = 1
    # prob[7, 10] = 1
    # prob[8, 10] = 1
    # prob[9, 1] = 1
    # prob[10, 1] = 1
    # prob[11, 10] = 1

    # joint training
    # prob = np.zeros([N, N])
    # prob[0, 10] = 1
    # prob[1, [9, 10]] = [0.98, 0.02]
    # prob[2, [9, 10]] = [0.31, 0.69]
    # prob[3, 1] = 1
    # prob[4, [0, 10]] = [0.45, 0.55]
    # prob[5, 10] = 1
    # prob[6, 10] = 1
    # prob[7, [9, 10]] = [0.18, 0.82]
    # prob[8, 0] = 1
    # prob[9, 1] = 1
    # prob[10, 0] = 1
    # prob[11, 9] = 1

    prob = np.zeros([N, N])
    prob[0, [1, 4, 7, 10]] = [0.59, 0.13, 0.01, 0.26]
    prob[1, [4, 7, 9, 10]] = [0.19, 0.04, 0.01, 0.74]
    prob[2, [1, 4, 7, 9, 10]] = [0.51, 0.11, 0.01, 0.01, 0.36]
    prob[3, [1, 4, 7, 10]] = [0.64, 0.1, 0.03, 0.22]
    prob[4, [1, 3, 7, 9, 10]] = [0.64, 0.01, 0.12, 0.03, 0.22]
    prob[5, [1, 3, 7, 10]] = [0.53, 0.10, 0.03, 0.36]
    prob[6, [1, 4, 7, 10]] = [0.58, 0.12, 0.01, 0.29]
    prob[7, [1, 4, 9, 10]] = [0.49, 0.12, 0.01, 0.32]
    prob[8, [1, 4, 7, 9, 10]] = [0.53, 0.10, 0.01, 0.01, 0.35]
    prob[9, [1, 4, 7, 10]] = [0.62, 0.11, 0.04, 0.23]
    prob[10, [1, 3, 4, 7, 9, 11]] = [0.69, 0.01, 0.18, 0.10, 0.01, 0.01]
    prob[11, [1, 4, 7, 10]] = [0.53, 0.11, 0.01, 0.35]

    plt.figure(figsize=(5, 5))
    plt.imshow(prob, cmap='Blues')
    plt.xticks(np.arange(N), np.arange(N) + 1)
    plt.xlabel('second view')
    plt.yticks(np.arange(N), np.arange(N) + 1)
    plt.ylabel('initial view')
    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = plt.text(j, i, prob[i, j], ha="center", va="center", )
    plt.tight_layout()
    plt.show()
    pass
