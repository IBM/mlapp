from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt


def _generate_confusion_matrix(y_true, y_pred, normalize):
    # Utility to generate the data used for the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def _plot_confusion_matrix(cm, ax, classes, title, normalize, cmap):
    # Utility to generate one confusion matrix image
    im = ax.matshow(cm, interpolation='nearest', cmap=cmap)
    ax.xaxis.set_ticks_position('bottom')
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticklabels=['']+classes.tolist(), yticklabels=['']+classes.tolist())
    ax.set_aspect('auto')
    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel('Predicted label', fontsize=6)
    ax.set_ylabel('True label', fontsize=6)  # relative to plt.rcParams['font.size']

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.margins(0.2)  # added to fix graph formatting
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    return ax


def plot_multiple_confusion_matrix(y_hat_train, y_train, y_hat_test, y_test):
    """
    This function will plot the confusion matrices for the train set and the test set, both as number and as proportion.
    :param model: trained instances of the model
    :param y_hat_train: pandas dataframe of the train set
    :param y_train: pandas series of the target related to the train set
    :param y_hat_test: pandas dataframe of the test set
    :param y_test: pandas series of the target related to the train set
    :return: One figure containing 4 images: confusion matrix for train set as number, confusion matrix for train set as
    frequencies, confusion matrix for test set as number, confusion matrix for test set as frequencies
    """
    fig, axes = plt.subplots(2, 2)

    titles = ['Train Confusion Matrix, without Normalization', 'Normalized Train Confusion Matrix',
              'Test Confusion Matrix, without Normalization', 'Normalized Test Confusion Matrix']

    norm_cm_train = _generate_confusion_matrix(y_train, y_hat_train, normalize=True)
    no_norm_cm_train = _generate_confusion_matrix(y_train, y_hat_train, normalize=False)
    norm_cm_test = _generate_confusion_matrix(y_test, y_hat_test, normalize=True)
    no_norm_cm_test = _generate_confusion_matrix(y_test, y_hat_test, normalize=False)

    train_classes = unique_labels(y_train, y_hat_train)
    test_classes = unique_labels(y_test, y_hat_test)

    axes[0, 0] = _plot_confusion_matrix(no_norm_cm_train, axes[0, 0], train_classes, titles[0], False, plt.cm.Blues)
    axes[0, 1] = _plot_confusion_matrix(norm_cm_train, axes[0, 1], train_classes, titles[1], True, plt.cm.Blues)
    axes[1, 0] = _plot_confusion_matrix(no_norm_cm_test, axes[1, 0], test_classes, titles[2], False, plt.cm.Blues)
    axes[1, 1] = _plot_confusion_matrix(norm_cm_test, axes[1, 1], test_classes, titles[3], True, plt.cm.Blues)

    fig.tight_layout()
    return fig
