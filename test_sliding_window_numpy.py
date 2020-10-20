import numpy as np
import cv2
import pyvmmonitor


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    # window length for sliding along the first axis
    L = 2
    s0, s1 = a.strides
    shape = a.shape
    out_shape = int(shape[0] / L), L, shape[1]
    strided = np.lib.stride_tricks.as_strided
    out = strided(a, shape=out_shape, strides=(s0, s0, s1))
    return out


@pyvmmonitor.profile_method
def strided_v2(a):
    sz = a.itemsize

    h, w = a.shape
    bh, bw = 3, 2
    shape = (int(h / bh), int(w / bw), bh, bw)

    strides = sz * np.array([w * bh, bw, w, 1])

    blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    shape = blocks.shape
    slide2d_v2_1 = blocks.reshape(
        (shape[0] * shape[1], shape[2], shape[3]))
    return slide2d_v2_1, original


def strided_v3(a):
    sz = a.itemsize
    h, w = a.shape
    bh, bw = 2, 1
    shape = (int(h / bh), bh, bw)

    strides = sz * np.array([w * bh, w, 1])

    blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return blocks


if __name__ == '__main__':
    im = cv2.imread('/home/visiona/Pictures/others/Lena.jpg')
    array_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    array_2d_test = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16],
                              [17, 18, 19, 20],
                              [21, 22, 23, 24]])
    array_2d_test_orig = np.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 11, 12],
                                   [13, 14, 15],
                                   [16, 17, 18]])
    # slide1d = strided_app(test_array, L=5, S=3)
    # slide2d = strided_app(array_2d_test_orig, L=2, S=0)
    slide2d_v2, original = strided_v2(array_2d_test)
    print(original)
    print(slide2d_v2)
    # print(slide2d_v2_1.shape)
    # slide2d_v3 = strided_v3(array_2d_test)
    # print(slide2d_v3)
