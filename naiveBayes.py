#naiveBayes.py
# -------------
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


import classificationMethod
import math

import dataClassifier

class LabelCounter:
    def __init__(self, legalLabels):
        if len(legalLabels) == 0:
            raise ValueError("legalLabels cannot be empty.")
        self.legalLabels = legalLabels

        self.labelCounts = {label: 0 for label in self.legalLabels}

    def increment_label(self, label):
        """
        Increment the count for the specified label if it exists in legalLabels.
        """
        if label in self.labelCounts:
            self.labelCounts[label] += 1
        else:
            raise ValueError(f"Label {label} is not in legalLabels")

    def get_num_labels(self):
        """
        Return the number of legal labels.
        """
        return len(self.legalLabels)

    def get_sum_counts(self):
        """
        Return the sum of all the counts for the labels.
        """
        return sum(self.labelCounts.values())  # Sum up all values in labelCounts


    def get_normalized_counts(self):
        """
        Return a dictionary where each label count is divided by the total sum of counts.
        This gives the relative probability of each label.
        """
        total = self.get_sum_counts()
        if total == 0:
            return {label: 0 for label in self.legalLabels}  # To avoid division by zero

        # Normalize each count by dividing by the total sum
        return {label: count / total for label, count in self.labelCounts.items()}

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    what we have in order to make the Bayes Model:
        - array of trainingData formatted
        - we need to sum them up label them with the classifiers
        - also need to get an idea of the prior probability
    what we will have to do after
        - make a method that takes in a Datum
        - calculate the probablity for label
        - take the max of the prob and make that your guess
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  #smoothing paramete
        self.automaticTuning = False  #flag to decide whether to choose k automatically ** use this in your train method **





    def train(self, trainingData, trainingLabels, validationData, validationLabels, dataset):


        if dataset == 1:
            legal_labels = range(0, 10)
            counter = LabelCounter(legalLabels=legal_labels)
        if dataset == 0:
            legal_labels = range(0, 2)
            counter = LabelCounter(legalLabels=legal_labels)
        for label in trainingLabels:
            counter.increment_label(label)

        self.prior_prob_table = counter.get_normalized_counts()

        self.conditional_prob_table = {label: [0] * len(trainingData[0]) for label in legal_labels}

        for feature_vector, label in zip(trainingData, trainingLabels):
            for feature_index, feature_value in enumerate(feature_vector):
                if feature_value > 0:  #which features are active and count them up
                    self.conditional_prob_table[label][feature_index] += 1

        for label in legal_labels:
            total_count = sum(self.conditional_prob_table[label]) + self.k * len(trainingData[0])
            self.conditional_prob_table[label] = [
                (count + self.k) / total_count for count in self.conditional_prob_table[label]
            ]

        return

    def classify(self, testData):
        """
        Classify the test data and return the most probable label for each data point.
        """
        predictions = []

        #for each feature vector in test data
        for feature_vector in testData:
            log_probabilities = {}  #log-probabilities for each label

            #log-probabilities for each possible label
            for label in self.prior_prob_table.keys():

                log_probabilities[label] = math.log(self.prior_prob_table[label])

                for feature_index, feature_value in enumerate(feature_vector):
                    if feature_value > 0:  #active feature
                        log_probabilities[label] += math.log(self.conditional_prob_table[label][feature_index])
                    else:
                        log_probabilities[label] += math.log(1 - self.conditional_prob_table[label][feature_index])

            #choose label with the highest log-probability
            best_label = max(log_probabilities, key=log_probabilities.get)
            predictions.append(best_label)

        return predictions

















