import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from encoder import Encoder

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

# Train model using the specified classifier
clf = RandomForestClassifier()
clf.fit(x, y)

# Split training data and test the accuracy of the model
y = train_data['has_new_comments']
x_train, x_test, y_train, y_test = train_test_split(train_data.loc[:, train_data.columns != 'has_new_comments'], y, test_size=0.25)
predictions = clf.predict(x_test)
accuracy = balanced_accuracy_score(predictions, y_test)
print('Accuracy: ', accuracy)

# Predict the has_new_comments column
x_test = test_data
predictions = clf.predict(x_test)
predictions = le.decode_column('has_new_comments', predictions)

# Format output to a csv file
output_data = {'ID': test_data.index, 'has_new_comments': predictions}
output = pd.DataFrame(output_data, columns=['ID', 'has_new_comments'])
output.to_csv('../solution.csv', index=False)