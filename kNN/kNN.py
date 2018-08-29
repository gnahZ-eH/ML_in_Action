from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt


def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels



def classfy0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# group, labels = creatDataSet()

# print classfy0([0,0], group, labels, 3)


def file2matrix(fileName):
    fr = open(fileName)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    resultMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        resultMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return resultMat, classLabelVector



# mat, lab = file2matrix('datingTestSet.txt')
#
# print mat
# print lab[0:20]



# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(mat[:, 0], mat[:, 1], 15.0*array(lab), 15.0*array(lab))
# plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVals, (m, 1))
    normalDataSet = normalDataSet / tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classsfierRes = classfy0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "predict = %d, real = %d" % (classsfierRes, datingLabels[i])
        if (classsfierRes != datingLabels[i]): errorCount += 1
    print "total rate = %f" % (errorCount / float(numTestVecs))


# datingClassTest()

def img2Vector(fileName):
    returnVect = zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# testVector = img2Vector('testDigits/0_13.txt')

# print testVector[0, 0:2]

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')

    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierRes = classfy0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "predict %d, real = %d" % (classifierRes, classNumStr)
        if (classifierRes != classNumStr): errorCount += 1.0

    print "total error = %d" % errorCount
    print "error rate = %f" % (errorCount / float(mTest))


handWritingClassTest()