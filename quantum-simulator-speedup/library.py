

from math import log2, pi
import math
import numpy as np
import pandas as pd

def angle180(angle_pi):
    angle = 180 * angle_pi / np.pi
    while angle > 360:
        angle -= 360
    while angle < 0:
        angle += 360
    return  float(angle)


def getExp(value):
    r = abs(value)
    if r == 0:
        return 0, 0
    cos, sin = value.real/r, value.imag/r
    # print(cos, sin)
    value /= r
    if value.real == 0:
        return round(float(r), 2), round(angle180(pi/2), 2) * (1 if sin > 0 else -1)
    # if value.imag == 0:
    #     return round(float(-r), 2),0  # 180和-180都是没有相位可以看做
    theta = math.atan(sin/cos)
    
#     大于0的时候就是对的了
    if cos < 0:
        if sin>0:
            theta = theta + pi
        else:
            theta = theta - pi

    # print(value, round(float(r), 2), round(angle180(theta), 2))
    return round(float(r), 2), round(angle180(theta), 2)


def complexString(value):
    exp_value = getExp(value)
    r, phase = exp_value
    if round(value, 4) == 0:
        return ''
    
    if phase == 0:
        return str(r)
    
    return '|'.join([str(r), str(round(phase),)])

def binaryString(intger, length = None):
    binary_string = bin(intger).replace('0b','')
    if length is None or length < len(binary_string):
        length = len(binary_string)
    length = int(length)
    return '0' * (length - len(binary_string)) + binary_string

def paresGateMatrixPd(matrix, global_phase = 0):
    length = matrix.shape[0]
    qubit_num = int(log2(length))
        
    data = {
        binaryString(qubit1, qubit_num): [0 for _ in range(length)]
        for qubit1 in range(length)
    }

    for row in range(length):
        ib = binaryString(row, qubit_num)
        for column in range(length):
            value = matrix[row][column]
            if abs(value) < 1e-5:
                data[ib][column] = ''
            else:
                data[ib][column] = complexString(value)
                # print(value, complexString(value))

    df=pd.DataFrame(data,index=[binaryString(qubit, qubit_num) for qubit in range(length)])
    return df
