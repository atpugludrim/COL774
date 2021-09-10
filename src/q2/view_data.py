import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',required=True)
    args = parser.parse_args()
    with open(args.f,'rb') as f:
        contents = np.load(f)

    # THIS IS SUPPOSED TO BE STOPPED BY PDB HERE
    print("A next line")
if __name__=="__main__":
    main()
