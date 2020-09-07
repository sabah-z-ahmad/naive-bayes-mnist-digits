# This program does the following:
#   1. Load MNIST data from pickle format
#   2. Extract features (mean and std. deviation)
#   3. Combine data with class labels
#   4. Sort data by class label
#   5. Split data by class
#   6. Save individual class data

import numpy as np
import pandas as pd
import pickle

NUM_DIGITS = 10

def main():
    # Import flattened test and train data (x) and associated labels (t)
    x_train, t_train, x_test, t_test = load()

    # Calculate average brightnesses and std. deviations for each sample
    train_features = get_mean_and_std(x_train)
    test_features = get_mean_and_std(x_test)

    # Load features and labels into dataframes
    df_train = get_dataframes(train_features, t_train)
    df_test = get_dataframes(test_features, t_test)

    # Sort dataframes by label in ascending order
    df_train = df_train.sort_values(by = ['label'])
    df_test = df_test.sort_values(by = ['label'])

    # Split train/test dataframe by class label into a collection of dataframes
    train_collection = split_by_label(df_train)
    test_collection = split_by_label(df_test)

    # Save each dataframe in the train/test collections as a separate pickle file
    save(train_collection, './data/train_')
    save(test_collection, './data/test_')


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


# Splits dataframe by class label returning a dictionary of the resulting dataframes
def split_by_label(df):
    dataframe_collection = {}
    for i in range(0, NUM_DIGITS):
        dataframe_collection[i] = df[df['label'] == i]

    return dataframe_collection


# Save each dataframe in a collection of dataframes as a separate pickle file
def save(dataframe_collection, prefix):
    for i in range(NUM_DIGITS):
        dataframe_collection[i].to_pickle(prefix + str(i) + '.pkl')


if __name__ == '__main__':
    main()
