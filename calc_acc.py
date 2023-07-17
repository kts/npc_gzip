"""
code for https://kenschutte.com/gzip-knn-paper/

This reads pre-computed data (distance matrix
and label files) from pickle files, {dumpdir}/{name}.pkl
and computes accuracy using different methods.

pkl file must contain a dict with fields:
 'test_label':  [int]
 'train_label': [int]
 'dis_matrix':  [[float]]

#compute all:
python calc_acc.py  --dumpdir DUMPDIR --seed 1 --save /tmp/results.json

should print something like,

           kinnews  kirnews  filipino swahili 
table5     0.891    0.905    0.998    0.927   
code       0.891    0.906    1.000    0.927   
top2       0.891    0.906    1.000    0.927   
knn1r      0.835    0.858    0.999    0.850   
knn1d      0.835    0.858    0.999    0.850   
knn2r      0.820    0.811    0.851    0.848   
knn3r      0.836    0.796    0.847    0.881   
knn2d      0.835    0.858    0.999    0.850   
knn3d      0.843    0.794    0.904    0.883


knn{k}{r|d} refers to knn with k value
and r|d is the method used for ties
(r=random, d=decrement)

see some options:
python calc_acc.py --help

"""
import os
import json
import pickle
import operator
import random
from collections import defaultdict,Counter
import numpy as np

#copy from Table 5 of the paper
tab5 = {
    'kinnews':  0.891,
    'kirnews':  0.905,
    'filipino': 0.998,
    'swahili':  0.927,
}

def calc_acc(
        dis_matrix,
        train_label,
        label,
        
        k = 2,
        rand = False,
):
    """
    This is a copy of the code of the 'calc_acc'
    method in experiments.py in npc_gzip repo.
    Changing as little as possible to ensure the same
    results as official repo.
    """
    correct = []
    pred = []

    compare_label = train_label
    start = 0
    end = k
        
    for i in range(len(dis_matrix)):
            sorted_idx = np.argsort(np.array(dis_matrix[i]))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(pred_labels.items(), key=operator.itemgetter(1), reverse=True)
            most_count = sorted_pred_lab[0][1]
            if_right = 0
            most_label = sorted_pred_lab[0][0]
            most_voted_labels = []
            for pair in sorted_pred_lab:
                if pair[1] < most_count:
                    break
                if not rand:
                    if pair[0] == label[i]:
                        if_right = 1
                        most_label = pair[0]
                else:
                    most_voted_labels.append(pair[0])
            if rand:
                most_label = random.choice(most_voted_labels)
                if_right = 1 if most_label==label[i] else 0
            pred.append(most_label)
            correct.append(if_right)

    return sum(correct)/len(correct)



def calc_acc_topk(D,
                  train_labels,
                  test_labels,
                  k = 2):
    """
    Compute top-k result from distance matrix, D.
    
    If any of the top-k results are the
    correct label, mark as correct.

    return accuracy (as fraction, 0-1)
    """
    correct = [] #:[bool]

    for i,row in enumerate(D):
        sorted_idx = np.argsort(row)
        labels = set((train_labels[j] for j in sorted_idx[:k]))
        correct.append(
            test_labels[i] in labels
        )

    return sum(correct) / len(correct)


def calc_acc_knn(D,
                 train_labels,
                 k = 2,
                 tie_method = 'rand',
                 ):
    """
    kNN classifier from distance matrix, D
    - D is shape (num_test,num_train)
    
    tie_method:
    - 'rand': use random.choice
    - 'dec':  on ties, continue to decrement 'k' until no ties.

    Not written for efficiency.

    return "hyp" (label hypotheses). array shape (num_test,)
    
    """
    assert(tie_method in (
        'rand',
        'dec',
    ))
    assert(k >= 1)

    def get_tied(sorted_idx):
        """
        return a list of labels that are all tied
        for highest count within sorted_idx.

        eg if there are now ties, output
        will be length 1.
        """
        labels = [train_labels[j] for j in sorted_idx]

        c = Counter(labels)
        (_,count), = c.most_common(1)

        return [k for (k,v) in c.items() if v == count]

    hyp = np.zeros((D.shape[0],), 'uint32')

    for i,row in enumerate(D):
        sorted_idx = np.argsort(row)

        tied_labels = get_tied(sorted_idx[:k])
        
        if tie_method == 'rand':
            hyp[i] = random.choice(tied_labels)
            
        else:
            assert(tie_method == 'dec')
            ## - decrease k until their are no ties
            ## - will always terminate because k=1 can't have ties

            kmod = k
            while len(tied_labels) > 1:
                kmod = kmod - 1
                tied_labels = get_tied(sorted_idx[:kmod])

            # not, it's length 1:
            hyp[i], = tied_labels

    return hyp


def calc_all(infile):
    """
    read one pickle dump from infile
    and compute all the results we want.

    returns dict: method_name => accuracy
    """
    d = pickle.load(open(infile,'rb'))

    test_labels  = np.array(d['test_label'])
    train_labels = np.array(d['train_label'])
    D            = np.array(d['dis_matrix'])

    num_train, = train_labels.shape
    num_test,  = test_labels.shape

    #sanity check:
    assert(D.shape == (num_test,
                       num_train))

    def to_acc(hyp):
        return (hyp == test_labels).mean()

    Lt = train_labels
    Ls = test_labels

    _knn = lambda k,tie_method : to_acc(calc_acc_knn(D,Lt, k=k,tie_method=tie_method))
    
    return dict([

        ('code',  calc_acc(D,Lt,Ls)),

        ('top2',  calc_acc_topk(D,Lt,Ls, k=2)),
        
        ('knn1r', _knn(1,'rand')),
        ('knn1d', _knn(1,'dec')),

        ('knn2r', _knn(2,'rand')),
        ('knn3r', _knn(3,'rand')),

        ('knn2d', _knn(2,'dec')),
        ('knn3d', _knn(3,'dec')),
    ])


def print_table(all_results):
    """
    print results table,
    all_results[name] = dict of name=>accuracy 
    """
    names = list(all_results.keys())

    #col widths
    c0 = 10
    c1 =  8

    l = ["".ljust(c0)] + [name.ljust(c1) for name in names]
    print(" ".join(l))

    keys = list(all_results[names[0]].keys())
    for key in keys:

        l = [key.ljust(c0)]
        for name in names:
            val = all_results[name][key]
            l.append(f"{val:.03f}".ljust(c1))

        print(" ".join(l))


def main():    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--names',
                        help='comma-sep dataset names OR "all"',
                        default='all')

    parser.add_argument('--seed',
                        help = 'start with random.seed(seed)',
                        type = int,
                        default = None)

    parser.add_argument('--dumpdir',
                        #required = True,
                        help='where to find $name.pkl dumps')

    parser.add_argument('--save',
                        default = None,
                        help='write computed results to json file')

    parser.add_argument('--load',
                        default = None,
                        help='load results from json and just print table')
    
    args = parser.parse_args()

    if not args.seed is None:
        random.seed(args.seed)

    if args.load:
        print("reading:", args.load)
        results = json.load(open(args.load))

    else:

        if args.names == 'all':
            # list of all names, get from tab5:
            names = list(tab5.keys())
        else:
            names = args.names.split(",")

        results = {}
        for name in names:
            infile = os.path.join(args.dumpdir, name + ".pkl")
            print("Reading:",infile)

            results[name] = {
                #first entry in table is just
                # read from 'tab5' constant:
                'table5': tab5[name],
            }
            results[name].update( calc_all(infile) )
             
    print_table(results)

    if args.save:
        json.dump(results,open(args.save,'w'))
        print("Wrote:",args.save)
    
        
if __name__ == "__main__":
    main()
