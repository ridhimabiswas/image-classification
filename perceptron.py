# perceptron.py
# ----------
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

import numpy as np

PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier with Averaged Perceptron implementation.
    """

    def __init__(self, legalLabels, feature_size):
        """
        Initializes the Perceptron classifier.

        Args:
        - legalLabels: A list of legal labels for classification.
        - feature_size: The size of the feature vectors
        """
        self.legalLabels = legalLabels #all the labels allowed
        self.type = "perceptron" #what type of classifier
        self.max_iterations = 20 #how many iterations
        self.feature_size = feature_size #how many feature vectors (784 for digits and 4200 for digits)
        self.weights = {label: np.zeros(self.feature_size) for label in self.legalLabels} #weights vector

    def setWeights(self, weights):
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels, dataset):
        """
        Train the perceptron on the provided data.

        """
        for _ in range(self.max_iterations):
            for feature_vector, label in zip(trainingData, trainingLabels):
                feature_vector = np.array(feature_vector)  #make sure feature_vector is a NumPy array

                #for each class we want to calculator a score
                scores = {lbl: np.dot(self.weights[lbl], feature_vector) for lbl in self.legalLabels}

                #predicted class is the max of the scores
                predicted_class = max(scores, key=scores.get)

                #update the weights if the predicted_class matched the true label or not
                if predicted_class != label:
                    self.weights[label] += feature_vector
                    self.weights[predicted_class] -= feature_vector

    def classify(self, data):
        """
        Classify each datum in 'data' by computing the dot product with the weights for each class
        and returning the label corresponding to the class with the highest score.

        Args:
        - data: List of feature vectors to classify.

        Returns:
        - predicted_labels: List of predicted labels for each datum.
        """
        predicted_labels = []

        for datum in data:
            #make sure the datum has the correct shape based on the feature size
            assert len(datum) == self.feature_size, f"Expected datum of length {self.feature_size}, but got {len(datum)}"

            #compute the scores for each label
            scores = {label: np.dot(self.weights[label], datum) for label in self.legalLabels}

            #pick the max of teh scores
            predicted_label = max(scores, key=scores.get)
            predicted_labels.append(predicted_label)

        return predicted_labels

