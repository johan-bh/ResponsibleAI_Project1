from sklearn.metrics import roc_curve, auc
import pandas as pd 
import matplotlib.pyplot as plt
from fairlearn.metrics import false_positive_rate,true_positive_rate


def plot_roc_simple(y_test, y_prob, a_test,y_pred,sensitive_attribute_name=None):
    df = pd.DataFrame({'y_test': y_test, 'y_prob': y_prob, 'a_test': a_test, 'y_pred': y_pred})

    # Plot ROC curves grouped by 'a_test'
    groups = df['a_test'].unique()
    plt.figure(figsize=(10, 8))

    for group in groups:
        group_data = df[df['a_test'] == group]
        fpr, tpr, _ = roc_curve(group_data['y_test'], group_data['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{group} (AUC = {roc_auc:.2f})')

    # Plotting the random chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # the point in the figure 
    for group in groups:
        group_data = df[df['a_test'] == group]
        plt.scatter(false_positive_rate(group_data['y_test'], group_data['y_pred']),
                    true_positive_rate(group_data['y_test'], group_data['y_pred']),
                    s=100, label=f'{group}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if sensitive_attribute_name:
        plt.title(f'ROC Curve grouped by {sensitive_attribute_name}')
    else: plt.title(f'ROC Curve grouped')
    plt.legend(loc="lower right")
    plt.show()