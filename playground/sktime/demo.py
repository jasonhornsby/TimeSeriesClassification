from sklearn import metrics
from sklearn.model_selection import train_test_split

from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_arrow_head, load_basic_motions


from sktime.classification.hybrid import HIVECOTEV1

X, y = load_arrow_head(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Create mrseql object
# use sax by default
ms = MrSEQLClassifier(seql_mode="clf")
# use sfa representations
ms2 = MrSEQLClassifier(seql_mode='fs', symrep=['sfa'])
# use sax and sfa representations
ms3 = MrSEQLClassifier(seql_mode='fs', symrep=['sax', 'sfa'])

# fit training data
ms.fit(X_train, y_train)
ms2.fit(X_train, y_train)
ms3.fit(X_train, y_train)

# prediction
predicted = ms.predict(X_test)
predicted2 = ms2.predict(X_test)
predicted3 = ms3.predict(X_test)

# Classification accuracy
print("Accuracy with mr-seql: %2.3f" %
      metrics.accuracy_score(y_test, predicted))

print("Accuracy with mr-seql: %2.3f" %
      metrics.accuracy_score(y_test, predicted2))

print("Accuracy with mr-seql: %2.3f" %
      metrics.accuracy_score(y_test, predicted3))
