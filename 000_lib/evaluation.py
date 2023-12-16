import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import classification_report
import pandas as pd

from tensorflow.keras import backend as K

def backend_f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def backend_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def backend_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def split_train_val_test(X, y, test_size=0.2, val_size=0.2, shuffle=True):
    X_neat, X_test, y_neat, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_neat, y_neat, test_size=val_size, shuffle=False)

    return X_train, y_train, X_val, y_val, X_test, y_test

def getConfusionMatrix(y_true, y_pred):
    return pd.crosstab(
        pd.Series(y_true, name='Actual'), 
        pd.Series(y_pred, name='Predicted')
    )

def calculateMetrics(confusion_m, minority_class=1):
    # To-do: return average metrics
    metrics = dict()
    sum_all = np.sum(confusion_m).astype('float')
    sum_diag = np.trace(confusion_m)
    metrics['accuracy'] = sum_diag / sum_all
    metrics['precision'] = confusion_m[minority_class, minority_class] / sum(confusion_m[:, minority_class])
    metrics['recall'] = confusion_m[minority_class, minority_class] / sum(confusion_m[minority_class, :])
    metrics['f1_score'] = metrics['precision'] * metrics['recall']   
    
    return metrics
    

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
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
    """
    

    
    metrics = calculateMetrics(cm)

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
    plt.xlabel('Predicted label\nAccuracy={:0.4f}'.format(
            metrics['accuracy']
        )
    )
    plt.show()
    
def evaluateClassificationNeuralNetwork(history, model, X_test, y_test, p=0.5, target_names=["no", "yes"] ,metrics=['accuracy', 'loss']):
    
    epochs = len(history["accuracy"])
    axis_x = range(1,epochs+1)
    
    plt.figure(1)
    
    n_rows = 1
    n_cols = 2
    if len(metrics) > 2:
        n_rows = 2
    
    for index, metric in enumerate(metrics):
        plt.subplot(n_rows, n_cols, index+1)
        
        plt.plot(axis_x, history[metric], label=metric)
        plt.plot(axis_x, history[f"val_{metric}"], label=f"val_{metric}")
        plt.legend()
    
    plt.show()
        
    plt.figure(2)    
    y_prediction = model.predict(X_test).reshape((1,len(X_test)))[0]
    y_prediction = list(map(lambda y: 1 if y > p else 0, y_prediction))
    confusion_m = getConfusionMatrix(y_test, y_prediction).to_numpy()
    plot_confusion_matrix(confusion_m, target_names)
    plt.show()
    
    print(classification_report(y_test, y_prediction, target_names=target_names))