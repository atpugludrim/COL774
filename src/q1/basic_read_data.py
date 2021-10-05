import re
import json
import string
import argparse
import matplotlib.pyplot as plt
import matplotlib

from q1a import parse, remove_punc
from q1d import dostemming, remove_stopwords
matplotlib.rcParams['text.usetex']=True
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',required=True,help='path to data')
    args = parser.parse_args()
    #############################################################
    lens = [[],[],[],[],[]]
    for i, l in enumerate(parse(args.p)):
        print(f"{i}",end="\r")
        stemmed = dostemming(remove_punc(remove_stopwords(l['reviewText']))).split()
        y = int(float(l['overall']))-1
        lens[y].append(len(stemmed))
    plt.boxplot(lens)
    plt.xticks([1,2,3,4,5],['$\\mathrm{Class \\;'+'{}'.format(k+1)+'}$' for k in range(5)])
    plt.ylabel('$\\mathrm{ Number\\; of\\; words\\; after\\; stemming\\; and\\; stopword\\; removal}$')
    plt.xlabel('$\\mathrm{Class}$')
    plt.savefig('freq_boxplot.png')
    #plt.show()

if __name__=="__main__":
    main()
