import numpy as np

def two_a(th, size, mu1, sig_sq1, mu2, sig_sq2, noise_sig_sq):
    size = int(size)
    x0 = np.ones(size)
    x1 = (np.random.randn(size)*np.sqrt(sig_sq1))+mu1
    x2 = (np.random.randn(size)*np.sqrt(sig_sq2))+mu2
    eps = np.random.randn(size)*np.sqrt(noise_sig_sq)
    ths = np.array(th+[1])
    xs = np.stack([x0,x1,x2,eps])
    # print(xs.shape)
    y = ths@xs
    # print(y.shape)
    return np.stack([x1,x2], axis = 1), y

def main():
    x, y = two_a([3.0,1.0,2.0], 1e6, 3.0, 4.0, -1.0, 4.0, 2.0)
    print(x, y)

if __name__ == "__main__":
    main()
