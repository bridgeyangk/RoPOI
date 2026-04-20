import numpy as np


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    return 1 if len(idx) != 0 else 0


def ndcg_last_timestep(y_true_seq, y_pred_seq, k):
    y_true = [y_true_seq[-1]]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    rel = 0
    for i in range(k):
        if top_k_rec[i] == y_true:
            rel = 1
            break
    dcg_k = rel / np.log2(i + 2)
    idcg_k = sum(1 / np.log2(i + 2) for i in range(min(k, len(y_true))))
    return dcg_k / idcg_k


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)
