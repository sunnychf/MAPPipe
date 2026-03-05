from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

support_models = {
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=0),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
}
