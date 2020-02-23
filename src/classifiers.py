from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

classifiers = {
  'MLP': MLPClassifier(),
  'KNN': KNeighborsClassifier(),
  'SVC': SVC(),
  'GP': GaussianProcessClassifier(),
  'DT': DecisionTreeClassifier(),
  'RF': RandomForestClassifier(),
  'AB': AdaBoostClassifier(),
  'GNB': GaussianNB(),
  'QDA': QuadraticDiscriminantAnalysis(),
  'GB': GradientBoostingClassifier(),
  'XGB': XGBClassifier()
}

def get_classifier(classifier):
  return classifiers.get(classifier)