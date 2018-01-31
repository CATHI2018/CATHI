import numpy as np

def calc_F1(traj_act, traj_rec, noloop=False):
    assert (isinstance(noloop, bool))
    assert (len(traj_act) > 0)
    assert (len(traj_rec) > 0)

    if noloop == True:
        intersize = len(set(traj_act) & set(traj_rec))
    else:
        match_tags = np.zeros(len(traj_act), dtype=np.bool)
        for poi in traj_rec:
            for j in range(len(traj_act)):
                if match_tags[j] == False and poi == traj_act[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(traj_act)
    precision = intersize * 1.0 / len(traj_rec)
    F1 = 2 * precision * recall * 1.0 / (precision + recall)
    return F1

def  calc_pairsF1(y, y_hat):
    assert (len(y) > 0)
    n = len(y)
    nr = len(y_hat)
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]: nc += 1


    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        F1 = 0
    else:
        F1 = 2. * precision * recall / (precision + recall)
    return float(F1)

def load(path):
    f = open(path,'r')
    recdict ={}
    real = []
    rec = []
    i = 0
    j = 0
    for line in f:
        if i%2 == 0:
            real.append(line)
        elif i%2 == 1:
            rec.append(line )
            recdict[j] ={'REAL':real[j],'REC':rec[j]}
            j +=1
        i+= 1
    return recdict



if __name__=="__main__":
    F1_tran = []; pF1_tran = []
    recdict_tran ={}
    recdict_tran = load('/results-path')
    for tid in sorted(recdict_tran.keys()):
        F1_tran.append(calc_F1(recdict_tran[tid]['REAL'], recdict_tran[tid]['REC']))

        pF1_tran.append(calc_pairsF1(recdict_tran[tid]['REAL'], recdict_tran[tid]['REC']))

    print('Tran  : F1 (%.3f, %.3f), pairsF1 (%.3f, %.3f)' %  (np.mean(F1_tran), np.std(F1_tran), np.mean(pF1_tran), np.std(pF1_tran)))
