import numpy as np
import pandas as pd
import pickle
import math


NUM_DIGITS = 10

def main():
    # Get the classes to classify from the user
    selected_classes = get_classes()
    n = len(selected_classes)

    # Load the train/test data for the selected digits
    df_train, df_test = load(selected_classes)

    # Obtain the mean and variance of each class for the classifier
    print("Constructing classifier...")
    df_classifier = df_train.groupby('label').agg(["mean", "var"])

    # Obtain the prior probabities for each class (0-9)
    priors = get_priors(df_train, selected_classes)


    # Make predictions on test data
    print("Making predicitons...")
    predictions = pd.DataFrame(np.zeros(df_test.shape[0]))
    for i in range(0, df_test.shape[0]):
        predictions.iloc[i] = predict(df_test.iloc[i], priors, df_classifier, selected_classes)

    # Calculate accuracy
    print("Calculating accuracy...")
    accuracies = calc_accuracies(predictions, df_test['label'], n)
    print_accuracies(accuracies, selected_classes)


def get_classes():
    possible_digits = []
    for i in range(0, NUM_DIGITS):
        possible_digits.append(str(i))

    # Ask user to select which digits they want to run classification on
    selected_digits = []
    load_all = None
    while load_all not in ("Y", "y", "N", "n"):
        load_all = input("Do you want to classify all ten digits? ")
        if load_all == "Y" or load_all == "y":
            for i in range(0, NUM_DIGITS):
                selected_digits.append(str(i))

        elif load_all == "N" or load_all == "n":
            selection = None
            while selection not in (possible_digits) and len(selected_digits) < 10:
                selection = input("Enter a digit to use (0-9) or X if finished: ")
                if selection in possible_digits:
                    selected_digits.append(selection)
                    possible_digits.remove(selection)

                elif selection == "X" or selection == "x":
                    if len(selected_digits) < 2:
                        print("You must enter two unique digits to perform classification.")
                    else:
                        break

                elif selection in selected_digits:
                    print("Invalid input. Please enter a unique digit or X if finished.")

                else:
                    print("Invalid input. Please enter a single digit between 0 and 9 inclusive.")

        else:
            print("Invalid input. Please enter Y of N.")

    return selected_digits


# Loads MNIST data saved in pickle format into numpy arrays
def load(selected_classes):
    train_collection = {}
    test_collection = {}

    for digit in selected_classes:
        train_collection[int(digit)] = pd.read_pickle("./data/train_" + digit + ".pkl")
        test_collection[int(digit)] = pd.read_pickle("./data/test_" + digit + ".pkl")

    df_train = pd.concat(train_collection, ignore_index = True)
    df_test = pd.concat(test_collection, ignore_index = True)

    return df_train, df_test


# Return the prior probabilities for each class based on frequency in the training set
def get_priors(df, selected_classes):
    # Get the total number of samples (rows)
    total = df.shape[0]

    # Count the number of samples in each class (0-9)
    counts = df['label'].value_counts().sort_index()

    # Calculate the prior probability for each class
    priors = np.zeros(NUM_DIGITS)

    i = 0
    for index in counts.index:
        priors[index] = counts.iloc[[i]] / total
        i += 1

#    for i in range(0, counts.shape[0]):
#        priors[i] = counts.iloc[[i]] / total

    return priors


# Predicts the label of a sample using Naive Bayes
def predict(sample, priors, df_classifier, selected_classes):
    posterior_probabilities = np.zeros(NUM_DIGITS)
    evidences = np.zeros(NUM_DIGITS)

    adj = 0
    for i in range(0, NUM_DIGITS):
        if str(i) in selected_classes:
            p_feature1_given_label = probability_distribution(sample.iloc[0], df_classifier.iloc[i+adj][0], df_classifier.iloc[i+adj][1])
            p_feature2_given_label = probability_distribution(sample.iloc[1], df_classifier.iloc[i+adj][2], df_classifier.iloc[i+adj][3])
            posterior_probabilities[i] = priors[i] * p_feature1_given_label * p_feature2_given_label
#            evidences[i] = p_feature1_given_label * p_feature2_given_label
        else:
            adj -= 1

#    evidence = 0
#    for i in range(0, len(evidences)):
#        evidence += priors[i] * evidences[i]

#    for i in range(0, len(posterior_probabilities)):
#        posterior_probabilities[i] = posterior_probabilities[i] / evidence

    return np.argmax(posterior_probabilities)


# Calculates the probability distribution
def probability_distribution(v, mean, var):
    return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-1 * ((v - mean)**2 / (2 * var)))


# Calculates the accuracies of the predicitons
def calc_accuracies(predictions, targets, n):
    counts = np.zeros(NUM_DIGITS)
    correct_predictions = np.zeros(NUM_DIGITS)

    for i in range(0, len(targets)):
        counts[targets.iloc[i]] += 1
        if(int(predictions.iloc[i]) == int(targets.iloc[i])):
            correct_predictions[targets.iloc[i]] += 1

    accuracies = np.zeros(NUM_DIGITS)
    for i in range(0, NUM_DIGITS):
        if(counts[i] != 0):
            accuracies[i] = correct_predictions[i] / counts[i]

    return accuracies

def print_accuracies(accuracies, selected_classes):
    for i in range(0, NUM_DIGITS):
        if str(i) in selected_classes:
            print(str(i) + ": " + str(accuracies[i]))



if __name__ == '__main__':
    main()
