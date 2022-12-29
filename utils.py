import matplotlib.pyplot as plt
import numpy as np
import itertools
 
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
 
    if cmap is None:
        cmap = plt.get_cmap('Blues')
 
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
 
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def plot_loss_acc_curve(train_loss_list, train_acc_list, dev_loss_list, dev_acc_list): 
    
    epochs = len(train_loss_list)
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot([e+1 for e in range(epochs)], train_loss_list, 'r', linewidth=2)
    ax1.plot([e+1 for e in range(epochs)], dev_loss_list, 'b', linewidth=2)
    plt.ylabel("CE loss")
    plt.xlabel("epoch")
    plt.ylim((0.1, 2))
    plt.xlim((0, epochs))
    plt.legend(["train", "dev"])
    plt.title("loss curves")
    plt.grid()

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot([e+1 for e in range(epochs)], train_acc_list, 'r', linewidth=1)
    ax2.plot([e+1 for e in range(epochs)], dev_acc_list, 'b', linewidth=1)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.ylim((0.4, 1))
    plt.xlim((0, epochs))
    plt.legend(["train", "dev"])
    plt.title("accurary curves")
    plt.grid()