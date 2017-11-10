# Matius Celcius Sinaga
# Ubuntu 14.0 | Python27

import numpy as np


def kdtree(pointList, depth=0):
    if not pointList:
        return

    #choose an axis based on the depth
    #memilih sebuah dasar sumbu yang terdalam
    k = len(pointList[0])
    ax = depth % k

    #choose an median along axis which to sort
    #memilih sebuah median sepanjang axis yang akan disortir/urutkan
    pointList.sort(key=lambda x: x[ax])
    median = len(pointList)/2

    #create kdtree iteratively
    #membuat kdtree secara berulang-ulang
    return [kdtree(pointList[0:median], depth+1),   # left subtree
            kdtree(pointList[median+1:], depth+1),  # right subtree
            pointList[median]]


def main():
    pointList = np.array([[2, 3],
                          [5, 4],
                          [9, 6],
                          [4, 7],
                          [8, 1],
                          [7, 2]])

    tree = kdtree(pointList.tolist())
    print tree


if __name__ == '__main__':
    main()
