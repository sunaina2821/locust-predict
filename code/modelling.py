import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    cohen_kappa_score,
)
import matplotlib


matplotlib.rcParams.update({"font.size": 16, "font.family": "Times New Roman"})
# load the dataset
df = pd.read_csv(
    "E:/Locust Breeding sites/locust_paper_data/locust_paper_data/preprocessed-data-without-nan.csv"
).dropna()

# extract the target variable and the features
X, y = (
    df.drop(
        [
            "presence",
            "Unnamed: 0",
            "Unnamed: 0.1",
            "x",
            "y",
            "method",
            "year",
            "month",
            "day",
            "observation_date",
        ],
        axis=1,
    ),
    df["presence"],
)

# label encode columns with string values

# standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=2
)

# create a logistic regression model with specified solver and regularization
model = LogisticRegression(
    solver="liblinear", penalty="l2", max_iter=1000, random_state=44
)

# fit the model to the training data
model.fit(X_train, y_train)

# predict the target variable for the testing set
y_pred = model.predict(X_test)

# evaluate the performance of the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Kappa score:", kappa)

# plot ROC curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# plot kappa score
plt.figure()
sns.heatmap([[kappa]], annot=True, cmap=plt.cm.Blues)
plt.title("Kappa Score")
plt.show()
