
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from os import listdir
from os.path import isfile, join
from scipy.ndimage import filters
import urllib
from hashlib import sha256
import math
import operator
import tensorflow as tf
from numpy import random

class Assignment: 
    """ main executions of all the parts in assignment 1"""
    def __init__(self):
        #setting subset of actors
        self.subsetActorsList =  ['butler', 'radcliffe', 'bartan', 'bracco', 'gilpin', 'harmon']
        self.subsetActorsToLabelsIndexDict =  {'butler':0, 'radcliffe':1, 'bartan':2, 'bracco':3, 'gilpin':4, 'harmon':5}
        self.totalFacesList = list(set([a.split("\t")[0] for a in open("allFaces.txt").readlines()]))
        self.trainingSet = []
        self.trainingLabels = []
        self.validationSet = []
        self.validationLabels = []
        self.testSet = []
        self.testLabels = []
        self.genderTestSet = []
        self.genderTestLabels = []

    def downloadUncroppedImages(self, actorList = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'], maxDownloadCount = 130):
        """
        Function: partZero
        Purpose: Download the images & cropped the faces
        Parameter: None
        Result: saved images
        """

        url = urllib.URLopener()            

        #Note: Use totalFacesList for all faces
        for actor in actorList:
            name = actor.split()[1].lower()
            i = 0
            for line in open("allFaces.txt"):
                if actor in line:
                    #download the file using threading for urllib.URLopener().retrieve
                    filename = name+"_"+str(i)+'.'+line.split()[4].split('.')[-1]
                    self.timeout(url.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                    if not os.path.isfile("uncropped/"+filename):
                        continue

                    #check the uncropped file has correct sha256
                    imageData = open("uncropped/"+filename).read()
                    hash_from_file = sha256(imageData).hexdigest()
                    if line.split()[6] != hash_from_file:
                        continue
                    
                    #read photo and read cropped color box
                    imageData = imread("uncropped/"+filename)
                    x1,y1,x2,y2 = line.split()[5].split(",")

                    if len(imageData.shape) == 3:
                        #cropped the photo
                        imageData = imageData[int(y1):int(y2), int(x1):int(x2), :]
                        #turn to gray scale
                        r, g, b = imageData[:,:,0], imageData[:,:,1], imageData[:,:,2]
                        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                        gray = gray/255
                    else:
                        imageData = imageData[int(y1):int(y2), int(x1):int(x2)]

                    #resize image to 32x32
                    processedImage = imresize(gray, (32, 32))
                    imsave("cropped/"+filename, processedImage)
                    #plt.imshow(processedImage, cmap = cm.Greys_r)
                    #plt.show()
                    
                    print filename
                    i += 1
                
                if i >= maxDownloadCount: 
                    break

    
    def getLabelsSubsetLabelsArray(self, label):
        result = zeros(6)
        result[self.subsetActorsToLabelsIndexDict[label]] = 1
        return np.asarray(result)


    def initializeInputImagesSets(self):
        #load training set, validation set, test set, and labels. 
        onlyfiles = [f for f in listdir("cropped/") if isfile(join("cropped/", f))]
        imageCountDict = {}

        for filename in onlyfiles: 
            #avoid reading .DS_Store file
            if filename[0] == ".":
                continue

            label = filename.split("_")[0]
            flattenImageData = imread("cropped/"+filename).flatten()

            if label in self.subsetActorsList: 
                if imageCountDict.has_key(label): 
                    imageCount = imageCountDict[label]
                    if imageCount <= 100:
                        self.trainingSet.append(flattenImageData/255.)
                        self.trainingLabels.append(self.getLabelsSubsetLabelsArray(label))
                    elif imageCount <= 110:
                        self.validationSet.append(flattenImageData/255.)
                        self.validationLabels.append(self.getLabelsSubsetLabelsArray(label))
                    elif imageCount <= 120:
                        self.testSet.append(flattenImageData/255.)
                        self.testLabels.append(self.getLabelsSubsetLabelsArray(label))
                    imageCountDict[label] += 1
                else: 
                    imageCountDict[label] = 1
            else: 
                if not imageCountDict.has_key(label):
                    self.genderTestSet.append(flattenImageData/255.)
                    self.genderTestLabels.append(label)
                    imageCountDict[label] = 1

        self.trainingSet = np.asarray(self.trainingSet)
        self.trainingLabels = np.asarray(self.trainingLabels)
        self.validationSet = np.asarray(self.validationSet)
        self.validationLabels = np.asarray(self.validationLabels)
        self.testSet = np.asarray(self.testSet)
        self.testLabels = np.asarray(self.testLabels)
        self.genderTestSet = np.asarray(self.genderTestSet)
        self.genderTestLabels = np.asarray(self.genderTestLabels)

    def timeout(self, func, args=(), kwargs={}, timeout_duration=1, default=None):
        import threading
        class InterruptableThread(threading.Thread):
            def __init__(self):
                threading.Thread.__init__(self)
                self.result = None

            def run(self):
                try:
                    self.result = func(*args, **kwargs)
                except:
                    self.result = default

        it = InterruptableThread()
        it.start()
        it.join(timeout_duration)
        if it.isAlive():
            return False
        else:
            return it.result

    def getSmallTrainingInput(self, inputSize):
        smallTrainingSet = []
        smallTrainingLabels = []
        
        imageCountDict =  {'butler':0, 'radcliffe':0, 'bartan':0, 'bracco':0, 'gilpin':0, 'harmon':0}
        indexToLastNameDict =  {0:'butler', 1:'radcliffe', 2:'bartan', 3:'bracco', 4:'gilpin', 5:'harmon'}

        trainingSize = len(self.trainingSet)
        randomIndexArray = random.permutation(trainingSize)
        maxSizePerPerson = inputSize/6
        for i in randomIndexArray:
            lastNameIndex = self.trainingLabels[i].tolist().index(1)
            lastName = indexToLastNameDict[lastNameIndex]
            if imageCountDict[lastName] < maxSizePerPerson:
                smallTrainingSet.append(self.trainingSet[i])
                smallTrainingLabels.append(self.trainingLabels[i])
                imageCountDict[lastName] += 1


        smallTrainingSet = np.asarray(smallTrainingSet)
        smallTrainingLabels = np.asarray(smallTrainingLabels)

        return smallTrainingSet, smallTrainingLabels

    def partOneGraphingResult(self):
        trainingLine, = plt.plot([0, 100, 300, 1000, 2100, 3200, 4600], [20, 38.6, 53.2, 64.4, 81.6, 87.2, 93.4], label='Training')
        validationLine, = plt.plot([0, 100, 300, 1000, 2100, 3200, 4600], [20, 42, 50, 60, 70, 80, 82], label='Validation')
        testLine, = plt.plot([0, 100, 300, 1000, 2100, 3200, 4600], [25, 45, 55, 57.5, 75, 77.5, 77.5], label='Test')
        plt.ylabel('Correctness(%)')
        plt.xlabel('Epoche Value')
        plt.legend(handles=[trainingLine, validationLine, testLine], loc=4)
        plt.show()

    def partOne(self, nhid=300):
        main.initializeInputImagesSets()

        x = tf.placeholder(tf.float32, [None, 1024])

        W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
        b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

        W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
        b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


        layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
        layer2 = tf.matmul(layer1, W1)+b1

        y = tf.nn.softmax(layer2)
        y_ = tf.placeholder(tf.float32, [None, 6])


        lam = 0.0001
        decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
        NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)

        train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(NLL)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        for i in range(5000):
            smallTrainingSet, smallTrainingLabels = self.getSmallTrainingInput(60)
            sess.run(train_step, feed_dict={x: smallTrainingSet, y_: smallTrainingLabels})

            if i % 100 == 0:
                print "i=",i
                print "Train:", sess.run(accuracy, feed_dict={x: self.trainingSet, y_: self.trainingLabels})
                print "Validation:", sess.run(accuracy, feed_dict={x: self.validationSet, y_: self.validationLabels})
                print "Test:", sess.run(accuracy, feed_dict={x: self.testSet, y_: self.testLabels})
                print "Penalty:", sess.run(decay_penalty)
        
        return W0.eval(sess)



    def partThree(self):
        
        # W300 = main.partOne(300)
        # #Code for displaying a feature from the weight matrix mW
        # fig = figure(1)
        # ax = fig.gca()
        # heatmap = ax.imshow(W300.T[0].reshape((32, 32)), cmap = cm.coolwarm)
        # fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        # show()

        W800 = main.partOne(800)
        #Code for displaying a feature from the weight matrix mW
        fig = figure(1)
        ax = fig.gca()
        heatmap = ax.imshow(W800.T[0].reshape((32, 32)), cmap = cm.coolwarm)
        fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        show()



if __name__ == "__main__":
    main = Assignment()
    #main.downloadUncroppedImages()
    #main.partOne()
    #main.partOneGraphingResult()
    main.partThree()

    