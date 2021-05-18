from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_test, y_pred, name=None):
    fig = plt.figure()
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    ax = plt.gca()
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(false_positive_rate, true_positive_rate, 'b', label=f'AUC = {roc_auc:.2f}')
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([-0.1, 1.05])
    ax.set_ylim([-0.1, 1.05])
    ax.grid()
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    if name:
        fig.savefig(name)
    return roc_auc, ax