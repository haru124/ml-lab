import nltk
nltk.download('names')
nltk.download('wordnet')
from sklearn.datasets import fetch_20newsgroups
groups = fetch_20newsgroups()
data_train = fetch_20newsgroups(subset='train', random_state=21)
train_label = data_train.target
data_test = fetch_20newsgroups(subset='test', random_state=21)
test_label = data_test.target
len(data_train.data), len(data_test.data), len(test_label)
import numpy as np
np.unique(test_label)
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names  # contains 5001 female names and 2943 male names
all_names = names.words()
WNL = WordNetLemmatizer()
def clean(data):
  cleaned = defaultdict(list)
  count = 0
  for group in data:
     for words in group.split():
        if words.isalpha() and words not in all_names:
            cleaned[count].append(WNL.lemmatize(words.lower()))
     cleaned[count] = ' '.join(cleaned[count])
     count +=1
  return(list(cleaned.values()))
x_train = clean(data_train.data)
x_test = clean(data_test.data)
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = tf.fit_transform(x_train)
X_test = tf.transform(x_test)
X_train.shape, X_test.shape

from sklearn.svm import SVC

#linear kernel (decision boundary will be a straight line/plane)
svc_lib = SVC(kernel='linear')

# C = Regularization parameter (controls margin vs. misclassification tradeoff)
parameters = {'C': (0.5, 1.0, 10, 100)}

# Import GridSearchCV for hyperparameter tuning with cross-validation
from sklearn.model_selection import GridSearchCV
import timeit

# - n_jobs = -1 → use all available CPU cores for faster computation
# - cv = 3 → 3-fold cross-validation (split training data into 3 parts)
grid_search1 = GridSearchCV(svc_lib, parameters, n_jobs=-1, cv=3)

start_time = timeit.default_timer()
grid_search1.fit(X_train, train_label)
final = timeit.default_timer() - start_time
print("Execution Time : ", final)

# Print the best parameter found (which value of C gave the best results)
print(grid_search1.best_params_)

# Print the best cross-validation accuracy score achieved during grid search
print(grid_search1.best_score_)

# Extract the best model (with the best 'C') from grid search
grid_search_best1 = grid_search1.best_estimator_

# Evaluate this best model on the test set
accur1 = grid_search_best1.score(X_test, test_label)
print(accur1)