import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          nticks=6):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # print color visualization
    color = np.zeros_like(cm)
    color = 255 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(color, interpolation='nearest', cmap=cmap, vmin=0, vmax=255)
    plt.title(title)

    ticks = np.arange(0, 256, 255 / (nticks - 1))
    acc_tick = np.linspace(0, 1, nticks)
    colorbar = plt.colorbar(ticks=ticks)
    colorbar.ax.set_yticklabels(acc_tick.astype('|S4'))

    # print accuracy number
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # custom layout and label
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45, ha='left')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
    target = [0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4,
              3, 3, 3, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
              0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4,
              3, 3, 3, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
              0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4,
              3, 3, 3, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
              0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4,
              3, 3, 3, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]

    output = [3, 0, 0, 1, 1, 1, 2, 2, 2, 9, 4, 4, 4,
              3, 3, 3, 5, 5, 5, 9, 9, 9, 0, 3, 3, 8, 9, 9, 9, 9, 9,
              8, 0, 6, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4,
              3, 6, 4, 0, 5, 5, 6, 6, 4, 1, 7, 1, 8, 8, 8, 9, 4, 6,
              0, 0, 0, 1, 1, 1, 5, 5, 4, 4, 4, 4, 4,
              1, 1, 0, 5, 5, 0, 0, 8, 0, 8, 0, 0, 8, 8, 8, 0, 0, 0,
              0, 0, 6, 7, 7, 7, 5, 5, 8, 9, 4, 4, 4,
              0, 0, 0, 0, 0, 5, 4, 6, 9, 7, 7, 7, 8, 4, 7, 9, 9, 9]

    classes = ["arranging_objects", "cleaning_objects", "having_meal",
               "microwaving_food", "making_cereal", "picking_objects",
               "stacking_objects", "taking_food", "taking_medicine",
               "unstacking_objects"]

    cfs_mat = confusion_matrix(target, output)

    # -----------------------
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cfs_mat, classes=classes, normalize=True,
                          title='Normalized confusion matrix',
                          cmap=plt.cm.Greens,
                          nticks=6)

    plt.show()
