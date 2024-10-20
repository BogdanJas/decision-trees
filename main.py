import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from chefboost import Chefboost as chef
import os

# Wine Quality dataset directly from a public URL
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(dataset_url, sep=';')

# Split dataset into features and target
if 'quality' not in df.columns:
    raise ValueError("The expected target column 'quality' was not found in the dataset.")

df['quality'] = df['quality'].apply(lambda x: 'Good' if x >= 6 else 'Bad').astype(str)

# Split the data into training and testing sets
X = df.drop(columns=['quality'])  # Features
y = df['quality']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Building a Decision Tree Classifier using CART
cart_clf = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_clf.fit(X_train, y_train)

y_pred_cart = cart_clf.predict(X_test)

print("Basic CART Decision Tree Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Classification Report for CART Decision Tree:")
print(classification_report(y_test, y_pred_cart))

plt.figure(figsize=(70, 50))
plot_tree(cart_clf, feature_names=X.columns, class_names=['Bad', 'Good'], filled=True, rounded=True, max_depth=3)
plt.title("CART Decision Tree Visualization")
plt.savefig('cart_decision_tree_visualization.png')
plt.close()

# 2. Building Decision Tree using ID3 and C4.5 with ChefBoost
# Prepare dataset for ChefBoost
chef_df = df.copy()
chef_df.columns = [str(i) for i in chef_df.columns]

output_dir = "outputs"
rules_dir = os.path.join(output_dir, "rules")
os.makedirs(rules_dir, exist_ok=True)

#ID3
config = {'algorithm': 'ID3', 'enableParallelism': False, 'output': output_dir}

try:
    id3_model = chef.fit(chef_df, config=config, target_label='quality')
except ImportError as e:
    print(f"Error occurred during ID3 training: {e}")
    print(f"Please check if the rules were generated correctly in {rules_dir}.")

id3_predictions = []
if 'id3_model' in locals():
    id3_predictions = [chef.predict(id3_model, list(X_test.iloc[i])) for i in range(len(X_test))]
    id3_predictions = ['Good' if pred == 'Good' else 'Bad' for pred in id3_predictions]
    print("ID3 Decision Tree Accuracy:", accuracy_score(y_test, id3_predictions))
    print("Classification Report for ID3 Decision Tree:")
    print(classification_report(y_test, id3_predictions))
else:
    print("ID3 model could not be created due to an error.")

#C4.5
config = {'algorithm': 'C4.5', 'enableParallelism': False, 'output': output_dir}
try:
    c45_model = chef.fit(chef_df, config=config, target_label='quality')
except ImportError as e:
    print(f"Error occurred during C4.5 training: {e}")
    print(f"Please check if the rules were generated correctly in {rules_dir}.")
c45_predictions = []
if 'c45_model' in locals():
    c45_predictions = [chef.predict(c45_model, list(X_test.iloc[i])) for i in range(len(X_test))]
    c45_predictions = ['Good' if pred == 'Good' else 'Bad' for pred in c45_predictions]
    print("C4.5 Decision Tree Accuracy:", accuracy_score(y_test, c45_predictions))
    print("Classification Report for C4.5 Decision Tree:")
    print(classification_report(y_test, c45_predictions))
else:
    print("C4.5 model could not be created due to an error.")

# 3. Tuning the Decision Tree Classifier for better performance
param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 10, 20],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=cart_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_cart_clf = grid_search.best_estimator_
y_pred_best = best_cart_clf.predict(X_test)
print("Tuned CART Decision Tree Accuracy:", accuracy_score(y_test, y_pred_best))
print("Best Parameters from Grid Search:", grid_search.best_params_)
print("Classification Report for Tuned CART Decision Tree:")
print(classification_report(y_test, y_pred_best))

feature_importances = best_cart_clf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), X.columns[indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Tuned CART Decision Tree")
plt.savefig('tuned_cart_decision_tree_feature_importance.png')
plt.close()

# 4. Comparing with Other Tree-Based Algorithms
# a) Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Plot feature importances for Random Forest
rf_importances = rf_clf.feature_importances_
indices = np.argsort(rf_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), rf_importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), X.columns[indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.savefig('random_forest_feature_importance.png')
plt.close()

# b) Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Classification Report for Gradient Boosting:")
print(classification_report(y_test, y_pred_gb))

# Plot feature importances for Gradient Boosting
gb_importances = gb_clf.feature_importances_
indices = np.argsort(gb_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), gb_importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), X.columns[indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Gradient Boosting")
plt.savefig('gradient_boosting_feature_importance.png')
plt.close()