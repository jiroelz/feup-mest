import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from encoder import Encoder
from classifiers import get_classifier

def is_posted_on_weekend(row):
  if (row.day_posted == 'SATURDAY' or row.day_posted == 'SUNDAY'):
    return 'yes'
  return 'no'

def is_current_day_weekend(row):
  if (row.current_day == 'SATURDAY' or row.current_day == 'SUNDAY'):
    return 'yes'
  return 'no'

def create_additional_columns(data):
  data['is_posted_on_weekend'] = data.apply(is_posted_on_weekend, axis=1)
  data['is_current_day_weekend'] = data.apply(is_current_day_weekend, axis=1)

  return data

le = Encoder()

# Read training data
train_data = pd.read_csv('../data/dev.csv', index_col='ID', na_values=':')
train_data = le.encode_dataframe(train_data)

# Read test data
test_data = pd.read_csv('../data/new.csv', index_col='ID', na_values=':')
test_data = test_data.drop('has_new_comments', axis=1)
test_data = le.encode_dataframe(test_data)

# Y = has_new_comments column
# X = everything else
x = train_data.drop('has_new_comments', axis=1)
y = train_data['has_new_comments']

# Split training data and test the accuracy of the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Train model using the specified classifier
clf = get_classifier('RF')
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
accuracy = balanced_accuracy_score(y_test, predictions)
print('Accuracy: ', accuracy)

# Predict the has_new_comments column
x_test = test_data
predictions = clf.predict(x_test)
predictions = le.decode_column('has_new_comments', predictions)

# Format output to a csv file
output_data = {'ID': test_data.index, 'has_new_comments': predictions}
output = pd.DataFrame(output_data, columns=['ID', 'has_new_comments'])
output.to_csv('../solution.csv', index=False)