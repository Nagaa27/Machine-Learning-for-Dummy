# Matius Celcius Sinaga
# Ubuntu 14.0 | Python27

import numpy as np
from MLP import MLP


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])

    #initialize
    #pengenalan diawal
    mlp = MLP(n_input_units=2, n_hidden_units=3, n_output_units=1)
    mlp.print_configuration()

    #training
    #melatih
    mlp.fit(inputs, targets)
    print 'Sedang melakukan pelatihan/training' #training
    print 'Bobot lapisan yang pertama: ' #first layer weight
    print mlp.v
    print 'Bobot lapisan yang kedua : ' #second layer weight
    print mlp.w

    # predict
    print 'Memprediksi ... ' #predict
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print i, mlp.predict(i)

if __name__ == '__main__':
    main()
