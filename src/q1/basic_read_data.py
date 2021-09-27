import re
import json
import string
import argparse
import matplotlib.pyplot as plt

def parse(path):
    with open(path,'r') as f:
        for l in f:
            yield json.loads(l)

punc = list(string.punctuation)
def remove_punc(text):
    global punc
    for p in punc:
        if p in text:
            text = text.replace(p,' ')
    text = re.sub("\\w*\\d+\\w*","",text)
    return text.strip().lower()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',required=True,help='path to data')
    parser.add_argument('-n',help='record number to show')
    parser.add_argument('-b',default=False,action='store_true',help='plot bar chart')
    parser.add_argument('-v',default=False,action='store_true',help='enumerate vocabulary')
    args = parser.parse_args()
    if args.n is not None:
        n = int(float(args.n))
    else:
        n = 15
    print("Showing record:",n)
    d = [0 for _ in range(5)]
    v = dict()
    #############################################################
    for i, l in enumerate(parse(args.p)):
        d[int(float(l['overall']))-1] += 1
        #
        if args.v:
            for t in remove_punc(l['reviewText']).split():
                if t not in v:
                    v[t] = 1
                else:
                    v[t] += 1
        #
        if i != n:
            continue
        for k in l:
            print("{}:\t{}".format(k,l[k]))
    print(len(v.keys()))
    if args.v:
        f = open('vocabulory.txt','w')
        for k in v:
            f.write("{}\t{}\n".format(k,v[k]))
        f.close()
    if args.b:
        fig, ax = plt.subplots(2,1)
        ax[0].bar([_ for _ in range(1,6)],d)
        ax[1].hist(v.values(),bins=100)
        plt.show()

if __name__=="__main__":
    main()
