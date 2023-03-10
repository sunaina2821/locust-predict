import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Stations, Daily, Point, Hourly

# Set time period
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)

# Get daily data   
'''
avg_temp = pd.DataFrame()
hoppers = pd.read_csv("Hoppers.csv")
hoppers = hoppers[hoppers['LOCRELIAB'] == 'Exact']
hoppers = hoppers[hoppers['COUNTRYID'].isin(['SU'])]
hoppers["STARTDATE"] = pd.to_datetime(hoppers["STARTDATE"], format="%Y-%m-%d %H:%M:%S")
hoppers['STARTDATE'] = pd.to_datetime(hoppers['STARTDATE']).dt.date
hoppers["STARTDATE"] = pd.to_datetime(hoppers["STARTDATE"], format="%Y-%m-%d")
'''
# avg_temp["avg_temp"] = hoppers.apply(lambda x: Daily(
#         Point(x['X'], x["Y"]),
#         x["STARTDATE"],
#         x["STARTDATE"],
#     )
#     .fetch()["tavg"],
#     axis=1,
# )
# print(avg_temp)

# def weather_stations_availability(X, Y):
    

processed_data_train = pd.read_csv("E:/Locust Breeding sites/locust_paper_data/locust_paper_data/train_val_ep_random.csv").dropna()
processed_data_test = pd.read_csv("E:/Locust Breeding sites/locust_paper_data/locust_paper_data/test_ep_random.csv").dropna()
print(processed_data_test, processed_data_train)

concat = pd.concat([processed_data_train, processed_data_test], axis=0)
concat.to_csv("E:/Locust Breeding sites/locust_paper_data/locust_paper_data/preprocessed-data-without-nan.csv")

non_temporal_columns = ['sand_0.5cm_mean', 'sand_5.15cm_mean']
temporal_variables = [
    'AvgSurfT_inst', 
    # 'Albedo_inst', 
    'SoilMoi0_10cm_inst', 
    'SoilMoi10_40cm_inst', 
    'SoilTMP0_10cm_inst', 
    'SoilTMP10_40cm_inst', 
    'Tveg_tavg', 
    'Wind_f_inst', 
    'Rainf_f_tavg', 
    'Tair_f_inst',
    'Qair_f_inst', 
    'Psurf_f_inst'
]
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

X, y = concat.drop(['presence', 'Unnamed: 0', 'Unnamed: 0.1', 'x', 'y', 'method', 'year', 'month' ,'day', 'observation_date'], axis=1), concat['presence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.34)
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=.34)

def evaluate(trues, preds, probs):
    labels = ['absence', 'presence']
    results = {}
    results['accuracy'] = metrics.accuracy_score(trues, preds)
    results['confusion_matrix'] = metrics.confusion_matrix(trues, preds)
    results['kappa'] = metrics.cohen_kappa_score(trues, preds)
    results['f1'] = metrics.f1_score(trues, preds)
    results['auc'] = metrics.roc_auc_score(trues, probs)
    
    print(metrics.classification_report(trues, preds, target_names=labels))
    return results

scaler = preprocessing.StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(train_x_scaled, train_y)
y_preds = model.predict(val_x_scaled)
y_probs = model.predict_proba(val_x_scaled)[:, 1]
logistic_val_results = evaluate(val_y, y_preds, y_probs)
y_preds = model.predict(test_x_scaled)
y_probs = model.predict_proba(test_x_scaled)[:, 1]
logistic_test_results = evaluate(y_test, y_preds, y_probs)

from sklearn.metrics import roc_curve

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    plt.show()

plot_roc_curve(y_test, y_probs)
