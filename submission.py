#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *
import time

SEED = time.time
############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    weights = dict(zip(['so', 'touching', 'quite', 'impressive', 'not', 'boring'], [1, 1, 0, 0, -1, -1]))
    return weights
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    words = x.split(' ')
    wordset = set(words)
    res = dict((word, words.count(word)) for word in wordset)
    return res
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    
    features = []
    beginTime = time.perf_counter()
    
    # Extracting feature vectors
    for key, value in trainExamples:
        features.append(featureExtractor(key))
        for word, _ in featureExtractor(key).items():
            weights[word] = value
    
    def loss(xFeature, y, w):
        if y == 1:
            p_w = sigmoid(dotProduct(xFeature, w))
        elif y == -1:
            p_w = 1 -sigmoid(dotProduct(xFeature, w))
        else:
            raise NotImplemented
        res = -math.log(p_w)
        return res
    
    # Measuring the gradient of loss according to the change of weight
    def gradLoss(w1, w2, index):
        feature = features[index]
        y = trainExamples[index][1]
        Dloss = loss(feature, y, w1) -loss(feature, y, w2)
        gradRes = dict(w1)
        increment(gradRes, -1, w2)
        for key, _ in gradRes.items():
            if gradRes[key] != 0:
                gradRes[key] = Dloss /gradRes.get(key, 0)
        return gradRes
    
    #Restruction needed
    def weightGen(w0):
        # Initializing
        wList = [dict(w0)]
        increment(w0, -eta, wList[0])
        while True:
            wNew = yield
            if wNew == None:
                break
            wList.append(dict(wNew))
            for index in range(len(trainExamples)):
                gLoss = gradLoss(wNew, wList[0], index)
                increment(wNew, -eta, gLoss)
            wList.pop(0)
        genRes = wNew
        return genRes
    
    def weightIter(num, w):
        gen = weightGen(weights)
        next(gen)
        for _ in range(num):
            gen.send(w)
        try:
            gen.send(None)
        except StopIteration:
            pass
        return
    
    weightIter(numIters, weights)
    print(time.perf_counter() -beginTime)
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_ANSWER

