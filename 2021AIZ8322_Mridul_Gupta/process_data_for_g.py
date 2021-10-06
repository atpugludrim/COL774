import re
import json
import time
import pickle
import string
import logging
import argparse
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

stop_words = set(stopwords.words('english'))
punkt = string.punctuation
ps = PorterStemmer()
def remove_stopwords(text):
    global stop_words
    splitstring = "\s|\.|,|\(|\)|-|\\\\|\"|!|#|\$|%|&|/|\*|\+|:|;|<|=|>|\?|@|\[|\]|\^|_|\`|\{|\}|\||~"
    words = [w.strip(punkt) for w in re.split(splitstring,text.strip())]
    new_words = []
    for w in words:
        if w.lower() not in stop_words:
            new_words.append(w)
    return " ".join(new_words)

def dostemming(text):
    global ps
    words = text.split()
    new_words = []
    for w in words:
        new_words.append(ps.stem(w))
    return " ".join(new_words)

def get_processed_line(line, stem_stop):
    return dostemming(remove_punc(remove_stopwords(line['summary'])))

def process_train(dpath,outpath,stem_stop,suff):
    #yf = open("{}_y.txt".format(outpath),'w')
    xf = open("{}_x.txt".format(outpath),'w')
    vocab = []
    for idx, line in enumerate(parse(dpath)):
        if not idx % 400:
            logging.info("Processing line {}".format(idx+1))
        y = int(float(line['overall']))
        x = get_processed_line(line,stem_stop)
        seen = []
        for word in x.split():
            if word not in seen:
                seen.append(word) # CACHE, NOT CHECKING FOR REPEATED WORDS IN VOCAB WHICH CAN BE HUGE
                if word not in vocab:
                    vocab.append(word)
        #
        #yf.write("{}\n".format(y))
        xf.write("{}\n".format(x))
    #yf.close()
    xf.close()
    vocab = np.sort(vocab).reshape(-1).tolist()
    with open("vocab_summary.txt","w") as f:
        for word in vocab:
            f.write("{}\n".format(word))

def process_test(dpath,outpath,stem_stop):
    #yf = open("{}_y.txt".format(outpath),'w')
    xf = open("{}_x.txt".format(outpath),'w')
    for idx, line in enumerate(parse(dpath)):
        if not idx % 400:
            logging.info("Processing line {}".format(idx+1))
        y = int(float(line['overall']))
        x = get_processed_line(line,stem_stop)
        #
        #yf.write("{}\n".format(y))
        xf.write("{}\n".format(x))
    #yf.close()
    xf.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dpath")
    parser.add_argument("-outpath")
    parser.add_argument("-stem-stop",default=False,action='store_true')
    parser.add_argument("-quiet",default=False,action='store_true')
    parser.add_argument("-write-vocab",default=False,action='store_true')
    args = parser.parse_args()
    ##############################################################
    # DO PREPROCESSING, SAVE VOCABULARY, SAVE WITHOUT CONVERTING #
    # TO NUMBERS, SORT VOCABULARY                                #
    ##############################################################
    
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
        
    suff = ""
    if args.stem_stop:
        suff += "_stem_stop"

    if args.write_vocab:
        process_train(args.dpath,args.outpath+suff,args.stem_stop,suff)
    else:
        process_test(args.dpath,args.outpath+suff,args.stem_stop)

if __name__=="__main__":
    main()
