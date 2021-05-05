from sklearn.metrics import f1_score


def get_auc(prob, labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rank_list = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    num_pos = 0
    num_neg = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            num_pos += 1
        else:
            num_neg += 1
    auc = 0
    auc = (sum(rank_list) - (num_pos * (num_pos + 1)) / 2) / (num_pos * num_neg)
    return auc


def pred(prob, threshold=0.5):
    res = []
    for p in prob:
        if p > threshold:
            res.append(1)
        else:
            res.append(0)
    return res


if __name__ == '__main__':
    # 1. AUC
    labels = [1, 0, 1, 1, 1, 0, 0, 0]
    prob1 = [0.7, 0.4, 0.3, 0.9, 0.45, 0.6, 0.5, 0.2]
    prob2 = [0.9, 0.1, 0.7, 0.3, 0.6, 0.2, 0.1, 0.8]
    print(get_auc(prob1, labels))
    print(get_auc(prob2, labels))

    # 2. F1
    pred1 = pred(prob1, 0.33)
    pred2 = pred(prob2, 0.5)
    print(f1_score(labels, pred1))
    print(f1_score(labels, pred2))

    print(pred1)
    print(pred2)