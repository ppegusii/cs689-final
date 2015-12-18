#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import gzip
import json
import os.path

def save(output, fname):
    with gzip.open(fname,'w') as f:
        json.dump(output, f)

def get_json(fname):
    return json.load(gzip.open(fname, 'r'))

def get_accuracy_std(data):
    #y_pred = np.array(np.concatenate(data['y_pred']))
    #y_true = np.array(np.concatenate(data['y_true']))
    y_pred = np.hstack(data['y_pred'])
    y_true = np.hstack(data['y_true'])

    std = np.std(np.array(data['acc'])*100.)
    acc = sum(y_pred == y_true) * 100. / len(y_pred)
    return acc, std


def get_data(house, f):
    SVM = get_json('../svm_models/output/svm_{house}{f}_all.json.gz'.format(house=house, f=f))
    CRF = get_json('../crf_models/output/crf_{house}{f}_all.json.gz'.format(house=house, f=f))
    SSVM = get_json('../ssvm_models/output/ssvm_{house}{f}_all.json.gz'.format(house=house, f=f))
    NB = get_json('../../results/nb/nb_{house}{f}.json.gz'.format(house=house, f=f))
    #HMM = get_json('../../results/hmm/hmm_{house}{f}.json.gz'.format(house=house, f=f))
    HMM = get_json('../hmm_model_f/hmm_{house}{f}_all.json.gz'.format(house=house, f=f))

    return SVM, CRF, SSVM, NB, HMM

def main():
    for house in ['A', 'B', 'C']:
        svm_arr = []
        crf_arr = []
        ssvm_arr = []
        nb_arr = []
        hmm_arr = []

        fname = ''+ house+'.json.gz'
        if os.path.exists(fname):
            [svm_arr, crf_arr, ssvm_arr, nb_arr, hmm_arr] = get_json(fname)
        else:
            for f in ['data', 'change', 'last']:
                svm, crf, ssvm, nb, hmm = get_data(house, f)
                res = map(lambda x: get_accuracy_std(x), [svm, crf, ssvm, nb, hmm])
                svm_arr.append(res[0])
                crf_arr.append(res[1])
                ssvm_arr.append(res[2])
                nb_arr.append(res[3])
                hmm_arr.append(res[4])

            save([svm_arr, crf_arr, ssvm_arr, nb_arr, hmm_arr],fname)
        draw_graph(house, svm_arr, crf_arr, ssvm_arr, nb_arr, hmm_arr)

def unpack(arr, index):
    return map(lambda x: x[index], arr)

def draw_graph(house, svm_arr, crf_arr, ssvm_arr, nb_arr, hmm_arr):
    width = 0.15       # the width of the bars
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind-2*width, unpack(hmm_arr,0), width, color='1.0', yerr=unpack(hmm_arr,1),
                    hatch=".",
                    error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

    rects2 = ax.bar(ind-width, unpack(crf_arr,0), width, color='0.25', yerr=unpack(crf_arr,1),
                    hatch="/",
                    error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

    rects3 = ax.bar(ind, unpack(svm_arr, 0), width, color='0.0', yerr=unpack(svm_arr, 1),
                    error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)
                    )
    rects4 = ax.bar(ind + width, unpack(ssvm_arr,0), width, color='0.5', yerr=unpack(ssvm_arr,1),
                    hatch="\\",
                    error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    rects5 = ax.bar(ind+2*width, unpack(nb_arr,0), width, color='0.75', yerr=unpack(nb_arr,1),
                    hatch="x",
                    error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Raw', 'Change', 'Last'))

    labels =  ('HMM','CRF', 'SVM', 'SSVM','Naive Bayes')
    rects = (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0])
    ax.legend(rects, labels, loc=3,ncol=2, mode="expand",
              bbox_to_anchor=(0., 1.02, 1., .102), borderaxespad=0.)

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)
    autolabel(ax, rects5)
    ax.set_ylim([0, 100])

    plt.tight_layout()

    plt.savefig(house + '.pdf', bbox_inches='tight')

    #plt.show()

def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


if __name__=="__main__":
    main()
