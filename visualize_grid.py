import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    precs = np.loadtxt(
        '/home/houyz/Code/MVselect/logs/wildtrack/resnet18_max_lr0.0005_b2_e10_dropcam0.0_2022-11-09_19-53-14/moda_90.7_Lstrategy65.1_Rstrategy67.6_theory70.8_avg61.0.txt')
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
    prob[0, 1] = 1
    prob[1, 0] = 1
    prob[2, 0] = 1
    prob[3, 2] = 1
    prob[4, 0] = 1
    prob[5, 1] = 1
    prob[6, 0] = 1

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
