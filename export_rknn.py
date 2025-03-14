import numpy as np
import cv2
from rknn.api import RKNN
import sys

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform='rk3588')

    ret = rknn.load_tflite(model=sys.argv[1])
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    print('--> Export rknn model')
    ret = rknn.export_rknn('./quant.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
