# import matplotlib
#
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, test_loss, train_result, test_result):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, title="loss")
    ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax1.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    ax1.legend()
    if train_result is not None and None not in train_result:
        ax2 = fig.add_subplot(122, title="result")
        ax2.plot(x_epoch, train_result, 'bo-', label='train' + ': {:.1f}'.format(train_result[-1]))
    else:
        ax2 = fig.add_subplot(122, title="result")
    ax2.plot(x_epoch, test_result, 'ro-', label='test' + ': {:.1f}'.format(test_result[-1]))
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)
