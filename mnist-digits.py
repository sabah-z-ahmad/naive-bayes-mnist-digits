import numpy as np
import pandas as pd
import pickle
import math

def main():

    # Import flattened test and train data (x) and associated labels (t)
    print("Importing data...")
    x_train, t_train, x_test, t_test = load()


    # Calculate average brightnesses and std. deviations for each sample
    print("Extracting features...")
    train_features = get_mean_and_std(x_train)
    test_features = get_mean_and_std(x_test)


    # Load features and labels into dataframes
    df_train = get_dataframes(train_features, t_train)


    # Sort dataframes by label in ascending order
    print("Sorting data...")
    df_train = df_train.sort_values(by = ['label'])


    # Obtain the mean and variance of each class for the classifier
    print("Constructing classifier...")
    df_classifier = df_train.groupby('label').agg(["mean", "var"])


    # Obtain the prior probabities for each class (0-9)
    priors = get_priors(df_train)


    # Make predictions on test data
    print("Making predicitons...")
    predictions = np.zeros(len(x_test))
#    for i in range(0, len(x_test)):
#        predictions[i] = predict(test_features[i], priors, df_classifier)
#    print(df_classifier)

    # Calculate accuracy
    print("Calculating accuracy...")
    accuracies = calc_accuracies(predictions, t_test)
    print(accuracies)



# Loads MNIST data saved in pickle format into numpy arrays
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


# Returns numpy array containing
#   Feature 1: The mean (average brightness)
#   Feature 2: The std. deviation
# for each sample
def get_mean_and_std(data):
    features = np.zeros([len(data), 2])
    i = 0
    for sample in data:
        features[i,0] = np.mean(sample)
        features[i,1] = np.std(sample)
        i += 1

    return features


# Converts and concatenates data and labels into a dataframe
def get_dataframes(data, labels):
    col_vals_data = ['feature1', 'feature2']
    col_vals_labels = ['label']

    df1 = pd.DataFrame(data, columns = col_vals_data)
    df2 = pd.DataFrame(labels, columns = col_vals_labels)

    return pd.concat([df1, df2], axis = 1)


# Return the prior probabilities for each class based on frequency in the training set
def get_priors(df):
    # Get the total number of samples (rows)
    total = df.shape[0]

    # Count the number of samples in each class (0-9)
    counts = df['label'].value_counts().sort_index()

    # Calculate the prior probability for each class
    priors = np.zeros(10)
    for i in range(0, counts.shape[0]):
        priors[i] = counts.iloc[[i]] / total

    return priors


# Predicts the label of a sample using Naive Bayes
def predict(sample, priors, df_classifier):
    posterior_probabilities = np.zeros(10)
    evidences = np.zeros(10)

    for i in range(0, len(posterior_probabilities)):
        p_feature1_given_label = probability_distribution(sample[0], df_classifier.iloc[i][0], df_classifier.iloc[i][1])
        p_feature2_given_label = probability_distribution(sample[1], df_classifier.iloc[i][2], df_classifier.iloc[i][3])
        posterior_probabilities[i] = priors[i] * p_feature1_given_label * p_feature2_given_label
#        evidences[i] = p_feature1_given_label * p_feature2_given_label

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
def calc_accuracies(predictions, targets):
    counts = np.zeros(10)
    correct_predictions = np.zeros(10)

    for i in range(0, len(targets)):
        counts[targets[i]] += 1
        if(int(predictions[i]) == int(targets[i])):
            correct_predictions[targets[i]] += 1

    accuracies = np.zeros(10)
    for i in range(0, len(accuracies)):
        accuracies[i] = correct_predictions[i] / counts[i]

    return accuracies



if __name__ == '__main__':
    main()
