import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from encoder import Encoder
import utils
from classifiers import get_classifier

# columns that should be checked for outliers
OUTLIER_COLS = []

# columns that should be standardized (z-score normalization)
STANDARDIZE_COLS = []

# columns that should be dropped
DROP_COLS = []

# columns on which a Log Transform should be applied
LOG_TRANSFORM_COLS = []

le = Encoder()

# Read training data
train_data = pd.read_csv('../data/dev.csv', index_col='ID', na_values=':')

train_data = utils.drop_cols(train_data, DROP_COLS)
train_data = utils.drop_missing_value_rows(train_data, 0.7)
train_data = utils.remove_outliers(train_data, OUTLIER_COLS)
train_data = utils.standardize(train_data, STANDARDIZE_COLS)
train_data = utils.log_transform(train_data, LOG_TRANSFORM_COLS)

train_data = le.encode_dataframe(train_data)

# Read test data
test_data = pd.read_csv('../data/new.csv', index_col='ID', na_values=':')
test_data = test_data.drop('has_new_comments', axis=1)

test_data = utils.drop_cols(test_data, DROP_COLS)
test_data = utils.drop_missing_value_rows(test_data, 0.7)
test_data = utils.remove_outliers(test_data, OUTLIER_COLS)
test_data = utils.standardize(test_data, STANDARDIZE_COLS)
test_data = utils.log_transform(test_data, LOG_TRANSFORM_COLS)

test_data = le.encode_dataframe(test_data)

# Y = has_new_comments column
# X = everything else
x = train_data.drop('has_new_comments', axis=1)
y = train_data['has_new_comments']

# Split training data and test the accuracy of the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Train model using the specified classifier
clf = get_classifier('XGB')
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: ', accuracy)

# Predict the has_new_comments column
x_test = test_data
predictions = clf.predict(x_test)
predictions = le.decode_column('has_new_comments', predictions)

# Format output to a csv file
output_data = {'ID': test_data.index, 'has_new_comments': predictions}
output = pd.DataFrame(output_data, columns=['ID', 'has_new_comments'])
output.to_csv('../solution.csv', index=False)