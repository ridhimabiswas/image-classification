# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



"""The code will be run through this file"""

import naiveBayes
import perceptron
import samples
import sys

import numpy as np
import statistics

TEST_SET_SIZE = 1000 #this size will change depending on face (150) and digits (1000)
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


"""following methods are to turn each datum (pixel representation of digit) into a feature vector"""
def countFeatures_D(one_datum):
    feature_vector = []  #empty list to store features

    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            #converting each pixel value into on or off
            if one_datum.getPixel(x, y) > 0:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

    return feature_vector  #now we have a complete feature vector for one image
def basicFeatureExtractorDigit(datums):
    vector = []
    for x in range(len(datums)):
        feature_vector = countFeatures_D(datums[x]) #extract feature from every image
        vector.append(feature_vector) #add this feature vector

    return vector



"""following methods are to turn each datum (pixel representation of face) into a feature vector"""
def countFeatures_F(one_datum):
    feature_vector = []  #empty list to store features

    width = one_datum.width
    height = one_datum.height

    for x in range(width):
        for y in range(height):
            #converting each pixel value into on or off
            if one_datum.getPixel(x, y) > 0:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

    return feature_vector  #now we have a complete feature vector for one image
def basicFeatureExtractorFace(datums):
    vector = []
    for x in range(len(datums)):
        feature_vector = countFeatures_F(datums[x]) #extract feature from every image
        vector.append(feature_vector)  #add this feature vector
    return vector


"""Reading command line argument"""
def default(str):
    return str + ' [Default: %default]'
USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """
def readCommand(argv):
    """Processing the command"""
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), #can choose if we want naivebayes or perceptron
                      choices=['naiveBayes', 'perceptron'],
                      default='naiveBayes')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], #faces or digits
                      default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=5000, type="int") #size will change depending on what set of images (5000 for digits or 451 for faces)
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int") #how much test data to be used


    options, otherjunk = parser.parse_args(argv) #ensure the right length of arguments and correct arguments are read
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    #set up variables like data, classifier, training set
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    print("training set size:\t" + str(options.training))

    if (options.data == "digits"):
        featureFunction = "digits"
        legalLabels = range(10)
        feature_size = 784
    elif (options.data == "faces"):
        featureFunction = "faces"
        legalLabels = [0, 1]
        feature_size = 4200
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)


    elif (options.classifier == "naiveBayes"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)

    elif (options.classifier == "perceptron"):
        classifier = perceptron.PerceptronClassifier(legalLabels, feature_size)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)


    args['classifier'] = classifier
    args['featureFunction'] = featureFunction

    return args, options



"""Running the actual classifier algorithm with feature vectors"""
def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']

    numTraining = options.training
    numTest = options.test


    #extract features
    print("Extracting features...")
    if featureFunction == "digits":
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                               DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
        dataset = 1

    if featureFunction == "faces":
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH,
                                               FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile("facedata/facedatavalidation", numTest, FACE_DATUM_WIDTH,
                                                 FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", numTest)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
        dataset = 0

    if featureFunction == "digits":
        trainingData = basicFeatureExtractorDigit(rawTrainingData)
        validationData = basicFeatureExtractorDigit(rawValidationData)
        testData = basicFeatureExtractorDigit(rawTestData)

    if featureFunction == "faces":
        trainingData = basicFeatureExtractorFace(rawTrainingData)
        validationData = basicFeatureExtractorFace(rawValidationData)
        testData = basicFeatureExtractorFace(rawTestData)

    #training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
def runClassifierWithRandomSubsets(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']


    numTraining = options.training
    numTest = options.test


    print("Extracting features...")
    if featureFunction == "digits":
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                               DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    elif featureFunction == "faces":
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH,
                                               FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)


    trainingDataSize = len(rawTrainingData)
    testDataSize = len(rawTestData)


    print(f"Training Data Size: {trainingDataSize}")
    print(f"Test Data Size: {testDataSize}")

    if featureFunction == "digits":
        trainingData = basicFeatureExtractorDigit(rawTrainingData)
        testData = basicFeatureExtractorDigit(rawTestData)
        dataset = 1
    elif featureFunction == "faces":
        trainingData = basicFeatureExtractorFace(rawTrainingData)
        testData = basicFeatureExtractorFace(rawTestData)
        dataset = 0

    acc = []

    #different percentage of training data
    import random
    import time

    acc = []
    total_training_time = 0
    #outer loop for subset percentages (10%, 20%, ..., 100%)
    for i in range(1, 11):  # For 10%, 20%, ..., 100% of training data
        training_subset_size = int(i * 0.1 * len(trainingData))
        accuracy_list = []

        for repeat in range(5):  #repeat 5 times for each percentage
            sampled_indices = random.sample(range(len(trainingData)), training_subset_size)
            sampled_training_data = [trainingData[j] for j in sampled_indices]
            sampled_training_labels = [trainingLabels[j] for j in sampled_indices]

            start_time = time.time()
            classifier.train(sampled_training_data, sampled_training_labels, testData, testLabels, dataset)
            training_time = time.time() - start_time
            total_training_time += training_time

            guesses = classifier.classify(testData)
            correct = sum(guesses[k] == testLabels[k] for k in range(len(testLabels)))
            accuracy = 100.0 * correct / len(testLabels)
            accuracy_list.append(accuracy)
            acc.append(accuracy)

            print(
                f"[{i * 10}% - Run {repeat + 1}] Training Time: {training_time:.2f}s | Accuracy: {accuracy:.1f}% | Error: {100 - accuracy:.1f}%")
        avg_training_time = total_training_time / 5

        mean_accuracy = statistics.mean(accuracy_list)
        mean_error = 100 - mean_accuracy  #calculate mean error
        std_dev_accuracy = statistics.stdev(accuracy_list)
        print(f"[{i * 10}%] Mean Accuracy: {mean_accuracy:.1f}% | Mean Error: {mean_error:.1f}% | "
              f"Std Dev: {std_dev_accuracy:.2f} | Avg Training Time: {avg_training_time:.2f}s\n")


    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    print(f"Overall Mean Accuracy: {mean_acc:.1f}% | Overall Std Dev: {std_acc:.1f}%")
    return mean_acc, std_acc



if __name__ == '__main__':
    #read input
    args, options = readCommand(sys.argv[1:])
    #run classifier
    #runClassifier(args, options)
    runClassifierWithRandomSubsets(args, options)


