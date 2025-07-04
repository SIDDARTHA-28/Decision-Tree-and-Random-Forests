# Decision-Tree-and-Random-Forests
Learn tree-based models for classification and regression
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# --- Classification: Iris dataset ---
iris = load_iris()
X_class, y_class = iris.data, iris.target
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# --- 1. Train a Decision Tree Classifier ---
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train_c, y_train_c)

# --- 2. Visualize the tree ---
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=iris.feature_names,
                           class_names=iris.target_names,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris_tree")

# --- 3. Analyze Overfitting by Varying Depth ---
train_acc, test_acc = [], []
for d in range(1, 10):
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train_c, y_train_c)
    train_acc.append(model.score(X_train_c, y_train_c))
    test_acc.append(model.score(X_test_c, y_test_c))

plt.plot(range(1, 10), train_acc, label='Train Accuracy')
plt.plot(range(1, 10), test_acc, label='Test Accuracy')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()
plt.show()

# --- 4. Train a Random Forest Classifier ---
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
rf_preds = rf_clf.predict(X_test_c)
print("Random Forest Accuracy:", accuracy_score(y_test_c, rf_preds))

# --- 5. Feature Importances ---
sns.barplot(x=rf_clf.feature_importances_, y=iris.feature_names)
plt.title("Feature Importances (Random Forest)")
plt.show()

# --- 6. Cross-validation ---
scores = cross_val_score(rf_clf, X_class, y_class, cv=5)
print("Cross-validation scores:", scores)
print("Average CV score:", np.mean(scores))

# --- Regression: California Housing ---
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# --- Decision Tree Regressor ---
dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_train_r, y_train_r)
y_pred_dt = dt_reg.predict(X_test_r)
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_dt)))

# --- Random Forest Regressor ---
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)
y_pred_rf = rf_reg.predict(X_test_r)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_rf)))
pip install scikit-learn matplotlib seaborn graphviz

