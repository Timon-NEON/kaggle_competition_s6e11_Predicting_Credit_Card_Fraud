import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, roc_auc_score


def roc_auc(y_val, predicted):
    score = roc_auc_score(y_val, predicted)
    print(f"ROC AUC: {score}")
    return score

def pr_auc(y_val, predicted):
    score = average_precision_score(y_val, predicted)
    print(f"PR AUC: {score}")
    return score

def full_score(y_val, predicted):
    score = dict()
    score['roc-auc'] = roc_auc(y_val, predicted)
    score['pr-auc'] = pr_auc(y_val, predicted)
    return score

def plot_pr_auc(y_val, predicted):
    ap = average_precision_score(y_val, predicted)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_val, predicted, ax=ax,
        name=f"AP={ap:.3f}",
        plot_chance_level=True,  # dashed baseline at prevalence
    )
    ax.set_ylim(0, 1)  # zoom in: precision is low here
    ax.set_title("Precision-recall (OOF)", loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02))
    plt.show()

def noise_check(y_val, predicted):
    boots = []
    rng = np.random.default_rng(42)
    y_true = np.asarray(y_val)
    for _ in range(1000):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].sum():
            boots.append(average_precision_score(y_true[i], predicted[i]))
    print(np.percentile(boots, [2.5, 97.5]))