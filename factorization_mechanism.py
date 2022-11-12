import math
from mpmath import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import bernoulli


def countSetBits(num):
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(num)

    # now separate out all 1's from binary string
    # we need to skip starting two characters
    # of binary string i.e; 0b
    setBits = [ones for ones in binary[2:] if ones == '1']

    return (len(setBits))


def computePrivacy(epsilon, delta):
    return (0.8888888 + 2 * math.log(1 / delta * math.sqrt(2 / math.pi))) ** (0.5) / epsilon


def matrixFactorization(repitition, streamlength, epsilon, delta):
    error_factorization = np.zeros(streamlength + 1)
    K = np.ones(streamlength)
    for i in range(streamlength):
        for j in range(i):
            K[i] *= (2 * j + 1) / (2 * (j + 1))

    Rfull = np.identity(streamlength)
    for j in range(streamlength):
        for k in range(j):
            Rfull[j, k] = K[j - k]

    for p in range(1, streamlength + 1):
        L = np.zeros(p)
        for i in range(p):
            L[i] = Rfull[p - 1, i]
        gaussian_vector = np.zeros(p)
        R = Rfull[:p, :p]
        for j in range(repitition):
            gaussian_vector += np.random.normal(0, 1, p)
        noise_vector = np.dot(R, gaussian_vector)
        error_factorization[p - 1] = np.dot(L, noise_vector)
        error_factorization[p - 1] = abs(error_factorization[p - 1]) * computePrivacy(epsilon, delta) / repitition

    return error_factorization


def binaryMechanism(repitition, streamlength, epsilon, delta):
    error_binary = np.zeros(streamlength + 1)
    for p in range(1, streamlength + 1):
        bit_ones = countSetBits(p)
        for j in range(repitition):
            for k in range(bit_ones):
                error_binary[p - 1] += np.random.normal(0, 1)
        error_binary[p - 1] = abs(error_binary[p - 1]) * privacy_term * computePrivacy(epsilon, delta) / repitition
    return error_binary


def computeMaxError(error):
    return la.norm(error, np.inf)


def bernoulliStream(probability, streamlength):
    stream = np.zeros(streamlength)
    X = bernoulli(0.5)
    stream = X.rvs(streamlength)
    count = np.zeros(streamlength + 1)
    count[0] = stream[0]
    for i in range(1, streamlength):
        count[i] = count[i - 1] + stream[i]

    return count


def uniformStream(universe, streamlength):
    stream = np.random.uniform(0, universe, streamlength)
    count = np.zeros(streamlength + 1)
    count[0] = stream[0]
    for i in range(1, streamlength):
        count[i] = count[i - 1] + stream[i]

    return count


repitition = 100000

# Run of the algorithms on all zero stream of length 2^16
streamlength = 2**16
epsilon = 0.5
delta = 1e-10
binary_count = binaryMechanism(repitition, streamlength, epsilon, delta)

ourbound_count = matrixFactorization(repitition, streamlength,epsilon,delta)


plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


# Run of the algorithm on stream with updates sampled using bernoulli trial
# with probability ranging from 0 until 0.9

for i in range(9):
    prob = i/10
    count = bernoulliStream(prob,streamlength)
    binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)

    ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

    plt.plot(count, label="Real running count bernoulli stream")
    plt.plot(binary_count, label="Binary tree mechanism")
    plt.plot(ourbound_count, label="Our mechanism")
    plt.xlim(1,streamlength-1)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("Absolute additive error")
    plt.show()

# Run of the algorithm on stream with updates picked from a uniform distribution
# over various range, ranging from 10 to 100

for i in range(10, 100, 10):
    count = uniformStream(i,streamlength)
    binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)

    ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

    plt.plot(count, label="Real running count uniform stream")
    plt.plot(binary_count, label="Binary tree mechanism")
    plt.plot(ourbound_count, label="Our mechanism")
    plt.xlim(1,streamlength-1)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("Absolute additive error")
    plt.show()




