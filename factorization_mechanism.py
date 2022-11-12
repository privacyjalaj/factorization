import math
from mpmath import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import bernoulli


def countSetBits(num):
    assert type(num) == int

    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(num)

    # now separate out all 1's from binary string
    # we need to skip starting two characters
    # of binary string i.e; 0b
    setBits = [ones for ones in binary[2:] if ones == '1']

    return (len(setBits))


def computePrivacy(epsilon, delta):
    '''
    :param epsilon: the privacy parameter, less than 1
    :param delta: the privacy parameter, ideally should be less than 1/streamlength
    :return: the privacy scaling required for Gaussian mechanism with sensitivity 1
    '''
    assert epsilon < 1
    assert delta < 1
    return (0.8888888 + 2 * math.log(1 / delta * math.sqrt(2 / math.pi))) ** (0.5) / epsilon


def matrixFactorization(repitition, streamlength, epsilon, delta):
    '''
    :param repitition: number of run of the factorization mechanism to get good confidence bound
    :param streamlength: the length of the stream
    :param epsilon: privacy parameter
    :param delta: privacy parameter
    :return: the output vector of the noise added on each coordinates as per factorization mech
    '''
    assert epsilon < 1
    assert delta < 1
    assert type(repitition) == int
    assert type(streamlength) == int

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
    '''
    :param repitition: number of run of the binary mechanism to get good confidence bound
    :param streamlength: total number of updates
    :param epsilon: privacy parameter
    :param delta: privacy parameter
    :return: the output vector of the noise added on each coordinates as per binary mech
    '''
    assert epsilon < 1
    assert delta < 1
    assert type(repitition) == int
    assert type(streamlength) == int

    error_binary = np.zeros(streamlength + 1)
    for p in range(1, streamlength + 1):
        bit_ones = countSetBits(p)
        for j in range(repitition):
            for k in range(bit_ones):
                error_binary[p - 1] += np.random.normal(0, 1)
        error_binary[p - 1] = abs(error_binary[p - 1]) * privacy_term * computePrivacy(epsilon, delta) / repitition
    return error_binary


def computeMaxError(error):
    '''
    :param error: the vector consisting of additive error per coordinate
    :return: the infinity norm of the error
    '''
    return la.norm(error, np.inf)


def bernoulliStream(probability, streamlength):
    '''
    :param probability: the probability with which an update is 1
    :param streamlength: the total number of updates
    :return: a stream of binary values with bit set to one with probability given
    by parameter probability
    '''
    assert probability <= 1
    assert probability >= 0
    assert type(streamlength) == int

    stream = np.zeros(streamlength)
    X = bernoulli(0.5)
    stream = X.rvs(streamlength)
    count = np.zeros(streamlength + 1)
    count[0] = stream[0]
    for i in range(1, streamlength):
        count[i] = count[i - 1] + stream[i]

    return count


def uniformStream(universe, streamlength):
    '''
    :param universe: the size of the universe from which uniform sample is picked
    :param streamlength: the total number of updates
    :return: returns a random number in the interval [1, universe]
    '''
    assert type(universe) == int
    assert type(streamlength) == int

    stream = np.random.uniform(0, universe, streamlength)
    count = np.zeros(streamlength + 1)
    count[0] = stream[0]
    for i in range(1, streamlength):
        count[i] = count[i - 1] + stream[i]

    return count



######### Code running the continual counting algorithms (our mechanism and binary mechanism) ##############

repitition = 1e6

'''
Run of the algorithms on all zero stream of length 2^16
The privacy parameter is chosen to be epsilon = 0.5 and delta = 1e-10
'''
streamlength = 2**16
epsilon = 0.5
delta = 1e-10

# Execution of the algorithm with all zero stream
prob = 0
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

# Plot the graphs for the real continual counts, that by factorization mechanism
# and that by the binary mechanism
plt.plot(count, label="Real running count with $p=0$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-4}
'''
prob = 2**(-4)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-4}
'''
plt.plot(count, label="Real running count with $p=2^{-4}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-5}
'''
prob = 2**(-5)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-5}
'''

plt.plot(count, label="Real running count with $p=2^{-5}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-6}
'''
prob = 2**(-6)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-6}
'''

plt.plot(count, label="Real running count with $p=2^{-6}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-7}
'''
prob = 2**(-7)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-7}
'''
plt.plot(count, label="Real running count with $p=2^{-7}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-8}
'''
prob = 2**(-8)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-8}
'''
plt.plot(count, label="Real running count with $p=2^{-8}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-9}
'''

prob = 2**(-9)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-9}
'''
plt.plot(count, label="Real running count with $p=2^{-9}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()


'''
Execution of the algorithm with stream with every update set to 1 with probability
equal to 2^{-10}
'''
prob = 2**(-10)
count = bernoulliStream(prob,streamlength)
binary_count = count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = count + matrixFactorization(repitition, streamlength,epsilon,delta)

'''
Plot the graphs for the real continual counts, that by 
factorization mechanism and that by the binary mechanism
when the probability of every update being 1 is 
2^{-10}
'''
plt.plot(count, label="Real running count with $p=2^{-10}$")
plt.plot(binary_count, label="Binary tree mechanism")
plt.plot(ourbound_count, label="Our mechanism")
plt.xlim(1,streamlength-1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()



######### Code running the continual histogram estimation using our mechanism and binary mechanism ##############

'''
Generate a random stream whose entry satisfy the 
Zipf Law. 
Zipf law has support over integer {1, 2, ..., k}
The following code is taken from 
https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html
 '''
a = 3.0
streamlength = 2048
s = np.random.zipf(a, streamlength)

from scipy.special import zeta
count = np.bincount(s)
k = np.arange(1, s.max() + 1)

plt.bar(k, count[1:], alpha=0.5, label='sample count')
plt.plot(k, streamlength*(k**-a)/zeta(a), 'k.-', alpha=0.5,
         label='expected count')
plt.semilogy()
plt.grid(alpha=0.2)
plt.legend()
plt.title(f'Zipf sample, a={a}, size={streamlength}')
plt.show()


'''
Create the stream of the input using Zipf's law. Here, we have the 
universe that is the size of the output produced earlier. 
    :param universe: size of the universe
    :param real_vector_max: compute the stream where we get maximum frequency element as input
    :param real_vector_max_running_count: compute the running count of the occurence of the max frequency
'''

universe = np.shape(count)
real_vector = np.zeros(universe)
real_vector_max = np.zeros(streamlength)
for i in range(streamlength):
    if (s[i]==1):
        real_vector_max[i] = 1

real_vector_max_running_count = np.zeros(streamlength+1)
real_vector_max_running_count[0] = real_vector_max[0]
for i in range(1,streamlength):
    real_vector_max_running_count[i] = real_vector_max_running_count[i-1] + real_vector_max[i]

binary_count = real_vector_max_running_count + binaryMechanism(repitition, streamlength, epsilon, delta)
ourbound_count = real_vector_max_running_count + matrixFactorization(repitition, streamlength,epsilon,delta)


'''
Plots the estimate of the maximum frequency through our algorithm (factorization mechanism)
and the estimate through the binary mechanism 
'''
plt.plot(real_vector_max_running_count, label="Highest frequency")
plt.plot(binary_count, label="Binary tree mechanism estimate")
plt.plot(ourbound_count, label="Our mechanism estimate")
#plt.yscale("log")
plt.xlim(10,streamlength)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Absolute additive error")
plt.show()

