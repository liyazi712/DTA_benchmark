from sklearn import metrics
from scipy import stats
import numpy as np
from math import sqrt
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy import stats

def eval_mse(y_true, y_pred, squared=True):
    """Evaluate mse/rmse and return the results.
    squared: bool, default=True
        If True returns MSE value, if False returns RMSE value.
    """
    return metrics.mean_squared_error(y_true, y_pred, squared=squared)
def eval_mae(y_true, y_pred):
    """Evaluate mae and return the result."""
    return metrics.mean_absolute_error(y_true, y_pred)

def eval_pearson(y_true, y_pred):
    """Evaluate Pearson correlation and return the results."""
    return stats.pearsonr(y_true, y_pred)[0]

def eval_ccc(y_true, y_pred):
    """
    Evaluate Concordance Correlation Coefficient (CCC). and return the results.
    """
    # 计算皮尔逊相关系数
    pearson_corr = eval_pearson(y_true, y_pred)
    mean_y_true = y_true.mean()
    mean_y_pred = y_pred.mean()
    numerator = 2 * pearson_corr - 1
    denominator = 1 - (mean_y_true - mean_y_pred) ** 2 / ((y_true - mean_y_true) ** 2).mean()
    ccc = numerator / denominator
    return ccc

def eval_spearman(y_true, y_pred):
    """Evaluate Spearman correlation and return the results."""
    return stats.spearmanr(y_true, y_pred)[0]

def eval_r2(y_true, y_pred):
    """Evaluate R2 and return the results."""
    return metrics.r2_score(y_true, y_pred)

def eval_auroc(y_true, y_pred):
    """Evaluate AUROC and return the results."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

def eval_auprc(y_true, y_pred):
    """Evaluate AUPRC and return the results."""
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(rec, pre)

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def rm2(y,f):
    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))


def evaluation_metrics(y_true=None, y_pred=None,
		eval_metrics=[]):
    """Evaluate eval_metrics and return the results.
    Parameters
    ----------
    y_true: true labels
    y_pred: predicted labels
    eval_metrics: a list of evaluation metrics
    """
    results = {}
    for m in eval_metrics:
        if m == 'mae':
            s = eval_mae(y_true, y_pred)
        elif m == 'mse':
            s = eval_mse(y_true, y_pred, squared=True)
        elif m == 'rmse':
            s = eval_mse(y_true, y_pred, squared=False)
        elif m == 'pearson':
            s = eval_pearson(y_true, y_pred)
        elif m == 'spearman':
            s = eval_spearman(y_true, y_pred)
        elif m == 'r2':
            s = eval_r2(y_true, y_pred)
        elif m == 'auroc':
            s = eval_auroc(y_true, y_pred)
        elif m == 'auprc':
            s = eval_auprc(y_true, y_pred)
        elif m == 'ci':
            s = ci(y_true, y_pred)
        elif m == 'rm2':
            s = rm2(y_true, y_pred)
        else:
            raise ValueError('Unknown evaluation metric: {}'.format(m))
        results[m] = s        
    return results