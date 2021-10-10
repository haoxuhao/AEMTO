import math
import numpy as np


def rosenblock(elements):
    obj = 0
    for i in range(len(elements) - 1):
        tmp_value = math.pow(elements[i], 2) - elements[i + 1]
        tmp_value1 = 100 * math.pow(tmp_value, 2)
        tmp_value2 = math.pow((elements[i] - 1), 2)
        obj += (tmp_value1 + tmp_value2)
    return obj


def ackley(elements):
    D = len(elements)
    obj = 20 + math.e - 20 * math.exp(-0.2 * math.sqrt(
        (1 / D) * sum([x**2 for x in elements]))) - math.exp(
            (1 / D) * sum([math.cos(2 * math.pi * x) for x in elements]))
    return obj


def rastrigin(elements):
    obj = sum([x**2 - 10*math.cos(2*math.pi*x) + 10 for x in elements])
    return obj


def griewank(elements):
    value1 = sum([x**2 / 4000 for x in elements])
    tmp = [math.cos(x / (math.sqrt(i + 1))) for i, x in enumerate(elements)]
    value2 = 1
    for e in tmp:
        value2 *= e
    return value1 - value2 + 1


def schwefel(elements):
    obj = 0
    sum_tmp = 0
    for i in range(len(elements)):
        sum_tmp += elements[i] * math.sin(math.sqrt(abs(elements[i])))
    obj = 418.9829 * len(elements) - sum_tmp
    return obj


def sphere(elements):
    obj = sum([x**2 for x in elements])
    return obj


if __name__ == "__main__":
    elements = [110.92369366114059, 117.23767669548637, 127.5855786846713, -6.7847721643051315, -107.54654850546936, -114.46653281949526, -137.73462070497487, -9.262727642878454, -91.60506139871518, 141.8239353758236]
    elements = [0] * 10
    f = 1205793353031677.8

    print("fitness: ", rastrigin(elements))
    print('ref fev', f)
