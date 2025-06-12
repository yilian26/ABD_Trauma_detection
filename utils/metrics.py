import torch
import numpy as np
import torch.nn as nn
from scipy.stats import sem, t, norm
from sklearn.utils import resample  
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn import metrics


def dice_score(preds, labels):
    """
    Compute the Dice Score.
    
    Parameters:
    preds (torch.Tensor): Predicted segmentation masks.
    labels (torch.Tensor): Ground truth segmentation masks.
    
    Returns:
    float: Dice Score.
    """
    # Ensure binary prediction
    preds = torch.tensor(preds > 0.5)
    labels = torch.tensor(labels > 0.5)
    
    intersection = (preds & labels).float().sum()  # Intersection points
    union = preds.float().sum() + labels.float().sum()  # Union points
    
    if union == 0:
        return torch.tensor(1.0)  # If both are zero, return perfect similarity
    else:
        dice = 2. * intersection / union
        return dice


def f1_score(predictions, true_labels):
    
    # Using argmax to determine predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    # True Positives, False Positives, False Negatives calculation using Numpy
    tp = (predicted_classes == 1) & (true_labels == 1)
    fp = (predicted_classes == 1) & (true_labels == 0)
    fn = (predicted_classes == 0) & (true_labels == 1)
    
    # Precision and Recall calculations
    tp_sum = np.sum(tp)
    precision = tp_sum / (np.sum(tp) + np.sum(fp) + 1e-8)
    recall = tp_sum / (np.sum(tp) + np.sum(fn) + 1e-8)

    # F1 Score calculation
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1

def wilson_binomial_confidence_interval(s, n, round_num=2, confidence_level=.95):
    '''
    Computes the binomial confidence interval of the probability of a success s, 
    based on the sample of n observations. The normal approximation is used,
    appropriate when n is equal to or greater than 30 observations.
    The confidence level is between 0 and 1, with default 0.95.
    Returns [p_estimate, interval_range, lower_bound, upper_bound].
    For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book.
    '''

    p_estimate = (1.0 * s) / n
    z = norm.interval(confidence_level)[1]
    
    wilson_p = (p_estimate + z**2/(2*n)) / (1 + z**2/n)
    
    wilson_interval_range = (z * np.sqrt( (p_estimate * (1-p_estimate))/n + z**2/(4*n**2) ) ) / (1 + z**2/n)
    
    interval_range =  z * np.sqrt( (p_estimate * (1-p_estimate))/n )
    output_p = f'%.{round_num}f'%(s/n)
    output_d = f'%.{round_num}f'%(wilson_p - wilson_interval_range)
    output_u = f'%.{round_num}f'%(wilson_p + wilson_interval_range)
    #return p_estimate, interval_range, p_estimate - interval_range, p_estimate + interval_range
    return f'{output_p}({output_d}-{output_u})'

def confusion_matrix_CI(tn, fp, fn, tp, round_num=2):
    acc = wilson_binomial_confidence_interval(tn+tp,tn+fp+fn+tp,round_num)
    PPV = wilson_binomial_confidence_interval(tp,tp+fp,round_num)
    NPV = wilson_binomial_confidence_interval(tn,fn+tn,round_num)
    Sensitivity = wilson_binomial_confidence_interval(tp,tp+fn,round_num)
    Specificity = wilson_binomial_confidence_interval(tn,fp+tn,round_num)
    return acc, PPV, NPV, Sensitivity, Specificity 


def confusion_matrix_CI_multi(cm, round_num=2):
    metrics = {}
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]

        acc = wilson_binomial_confidence_interval(tp + tn, tp + tn + fp + fn, round_num)
        PPV = wilson_binomial_confidence_interval(tp, tp + fp, round_num)
        NPV = wilson_binomial_confidence_interval(tn, tn + fn, round_num)
        Sensitivity = wilson_binomial_confidence_interval(tp, tp + fn, round_num)
        Specificity = wilson_binomial_confidence_interval(tn, tn + fp, round_num)

        metrics[i] = {
            'ACC': acc,
            'PPV': PPV,
            'NPV': NPV,
            'Sensitivity': Sensitivity,
            'Specificity': Specificity
        }
    return metrics


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    Youden_index = np.argmax(tpr - fpr)
    optimal_threshold = threshold[Youden_index]
    point = [fpr[Youden_index],tpr[Youden_index]]
    
    return optimal_threshold, point

def get_roc_CI(y_true, y_score):
#     roc_curves, auc_scores = zip(*Parallel(n_jobs=4)(delayed(bootstrap_func)(i, y_true, y_score) for i in range(1000)))
    roc_curves, auc_scores, aupr_scores = [], [], []
    for j in range(1000):
        yte_true_b, yte_pred_b = resample(y_true, y_score, replace=True, random_state=j)
        roc_curve_element = roc_curve(yte_true_b, yte_pred_b)
        auc_score = roc_auc_score(yte_true_b, yte_pred_b)
        aupr_score = auc(*metrics.precision_recall_curve(yte_true_b, yte_pred_b)[1::-1])

        roc_curves.append(roc_curve_element)
        auc_scores.append(auc_score)
        aupr_scores.append(aupr_score)

    #print('Test AUC: {:.3f}'.format(metrics.roc_auc_score(y_true, y_score)))
    #print('Test AUC: ({:.3f}, {:.3f}) percentile 95% CI'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))) 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fpr, tpr, _ in roc_curves:
        #print(scipy.interp(mean_fpr, fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(metrics.auc(fpr, tpr))
            
    mean_tpr = np.mean(tprs, axis=0)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    return roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper


def mean_and_confidence_interval(data):
    """
    Compute the mean and 95% confidence interval for a list of numbers.
    
    Parameters:
    data (list): List of dice scores.
    
    Returns:
    tuple: mean, lower bound of the CI, upper bound of the CI
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least two values for meaningful statistical analysis.")
    
    mean_dice = np.mean(data)
    confidence = 0.95
    error_margin = sem(data) * t.ppf((1 + confidence) / 2., len(data)-1)
    
    return mean_dice, mean_dice - error_margin, mean_dice + error_margin



