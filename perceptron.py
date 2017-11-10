# Matius Celcius Sinaga
# Ubuntu 14.0 | Python27

"""
This program will give you diffrent output result when you run it
Program ini akan menghasilkan hasi keluaran yang berbeda ketika anda menjalankanya
"""
import numpy as np
from pylab import rand, norm
import matplotlib.pyplot as plt
from itertools import repeat


class Perceptron(object):
    """
    Perceptron Class
	kelas perceptron
	"""

    def __init__(self, inputs, targets, T=6, eta=0.1):
        """
        T: number of iterations
        eta: learning rate
        N: number of training data
        m: number of inputs(exclude bias node)
        n: number of neurons
        w: m x n array
        inputs: N x m array
        targets: N x n array
        
		T 	= jumlah dari iterasi
		eta = nilai learning/pembelajaran
		N 	= jumlah dari data yang dilatih (training data)
		m 	= jumlah dari masukan (exclude bias node)
		n	= jumlah dari neuron
		w	= m x n array
		masukan/input = N x m array
		target = N x n array
		"""
        self._T = T
        self._eta = eta
        self._N = inputs.shape[0]
        self._m = inputs.shape[1]
        self._n = targets.shape[1]
        self.w = np.random.rand(self._m+1, self._n) * 0.1 - 0.05

        bias = - np.ones((self._N, 1))
        self._inputs = np.concatenate((bias, inputs), axis=1)
        self._targets = targets
        self._outputs = np.zeros((self._N, self._n))

        #Num of training data
        print 'Jumlah data yang dilatih: %d' % self._N 
        
        #Num of input dim
        print 'Jumlah nilai masukan dim.: %d' % self._m
        
        #Num of output dim
        print 'Jumlah dari keluaran dim.: %d' % self._n

    def fit(self):
        """
        a training phase
        Tahap pelatihan
        """
        for t in xrange(self._T):
            self._outputs = self.predict(self._inputs)
            self.w += self._eta * np.dot(self._inputs.T, self._targets - self._outputs)
        print 'Fase pelatihan' #training phase
        print 'bobot :' #weights:
        print self.w

    def predict(self, x):
        """
        activation function
        mengaktivasi fungsi
        x: N x m array
        w: m x n array
        """
        y = np.dot(x, self.w)
        return np.where(y > 0, 1, 0)

#on this section procces random(rand) result 
#lihat ini adalah salah satu bagian dalam program yang memberikan hasil rand/random/acak
def gen_data(n):
    xb = (rand(n)*2-1)/2-0.5
    yb = (rand(n)*2-1)/2+0.5
    inputs = [[xb[i], yb[i]] for i in xrange(len(xb))]
    targets = [[i] for i in repeat(1, len(xb))]

    xr = (rand(n)*2-1)/2+0.5
    yr = (rand(n)*2-1)/2-0.5
    inputs = inputs + [[xr[i], yr[i]] for i in xrange(len(xr))]
    targets = targets + [[i] for i in repeat(0, len(xr))]

    return np.array(inputs), np.array(targets)


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [0], [1]])

    p = Perceptron(inputs, targets)
    p.fit()

    print 'Tahap melakukan prediksi' #predict phase
    inputs_bias = np.concatenate((-np.ones((inputs.shape[0], 1)), inputs), axis=1)
    print p.predict(inputs_bias)

    print '\n'
    inputs2, targets2 = gen_data(20)
    p2 = Perceptron(inputs2, targets2)
    p2.fit()

    print '\n melakukan prediksi ...' #predict phase
    test_inputs2, test_targets2 = gen_data(10)
    test_inputs_bias2 = np.concatenate((-np.ones((test_inputs2.shape[0], 1)), test_inputs2), axis=1)
    print p2.predict(test_inputs_bias2)

    for i, x in enumerate(test_inputs2):
        if test_targets2[i][0] == 1:
            plt.plot(x[0], x[1], 'ob')
        else:
            plt.plot(x[0], x[1], 'or')

    n = norm(p2.w)
    ww = p2.w / n
    ww1 = [ww[1], -ww[0]]
    ww2 = [-ww[1], ww[0]]
    plt.plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
    plt.show()

if __name__ == '__main__':
    main()
