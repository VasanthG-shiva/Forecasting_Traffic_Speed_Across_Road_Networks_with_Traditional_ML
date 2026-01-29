#==================================================================================
#1.IMPORT LIBRARIES
#==================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from sklearn.cluster import KMeans


#===========================================================================================
df = pd.read_csv("PEMS-BAY.csv")
print(f'Rows and columns\n {df.shape}')
print(df.head())
'''Rename first columns name'''
df = df.rename(columns={"Unnamed: 0":"timestamp"})

'''convert timestamp + sort'''

df['timestamp'] = pd.to_datetime(df['timestamp'])#convert strings into real datetime
df= df.sort_values('timestamp')#Arrange rows by date (ascenting)

'''Set timestamp as index(time-series formet)'''

df = df.set_index('timestamp')#makes timestamp the index("we convert timestamp into the index so pandas can handle
                               #sorting,slicing,resampling and rolling opreations efficiently in time-series
'''Pick one sensor column '''
if len(df.columns)>0:
    col = df.columns[0]
    print("Using:",col)
    print(df.columns[:5])
#==================================================================================


#================================================================================
'''EDA SECTION'''
#===================================================================================
#Time series plot
plt.figure(figsize=(12,4))
plt.plot(df[col].iloc[:2000])
plt.title(f"Traffic Speed Time Series(Smaple window) - {col}")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.show()

#Hourly avearge
hourly_average = df[col].groupby(df.index.hour).mean()
plt.figure(figsize=(8,4))
plt.plot(hourly_average,marker ="o")
plt.title("Average Speed by Hour")
plt.xlabel("Hour of day")
plt.ylabel("Average Speed")
plt.grid(True)
plt.show()

#Day of week


day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
colors = ['red','red','red','red','red','green','green']
daily_avg = df[col].groupby(df.index.dayofweek).mean()

plt.figure(figsize=(8,4))
plt.bar(day_names, daily_avg.values, color=colors)
plt.xlabel("Day of Week")
plt.ylabel("Average Speed (km/h)")
plt.title("Average Speed by Day of Week")
plt.show()

#Speed distribution
print(df[col].head())
print(df[col].dtypes)
print("Non-NaN",df[col].dropna().shape[0])
plt.figure(figsize=(8,4))
plt.hist(df[col].dropna(),bins=50)
plt.title("Speed Distribution")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.show()

#Jam probability by hour
jam = df[col]<30
jam_by_hour = jam.groupby(df.index.hour).mean()
plt.figure(figsize=(8,4))
plt.plot(jam_by_hour)
plt.title("Jam probability by hour")
plt.xlabel("day hours")
plt.ylabel("jam")
plt.show()
#===========================================================================
'''Check missing values'''
#==============================================================================

print("Missig values:",df.isna().sum().sum())
print("Missig values:",df.isnull().sum().sum())

'''Remove impossible speeds'''

df = df.clip(lower=0,upper=120) #Remove Wrong senor readings

'''Final check'''

print(df.info())
print(df.head())

'''Now my dataset is 
*loaded
*Time column fixed
*Sorted
*Missing filled
*clean speed values
*Ready for feature engineering'''
#===============================================================================
'''Feature Engineering'''
#===============================================================================

'''Create lag & rolling features'''
#=======================================================================================
data = pd.DataFrame()                #create empty dataset
data['y'] = df[col]                  #Target = next speed
data['lag_1'] = df[col].shift(1)     #5 mins before speed
data['lag_6'] = df[col].shift(6)     #30 mins before speed
data['lag_12'] = df[col].shift(12)   #1hours before speed
data["rolling_mean_6"] = df[col].rolling(6).mean()
data["rolling_std_6"] = df[col].rolling(6).std()

'''Add time feature '''
data['hour'] = data.index.hour
data['day'] = data.index.dayofweek

'''Remove NaN rows'''
data = data.dropna()
print(data.head())
print(data.shape)
print(data.isna().sum())
'''
*Target
*Past Values
*Trend
*Hour
*Week pattern'''
#==============================================================================
'''Train-Test Split(Time-Based)'''
#===============================================================================
split_idx = int(len(data)*0.8)

train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

X_train = train.drop('y',axis=1)
y_train = train['y']

X_test = test.drop('y',axis=1)
y_test = test['y']

print("Train shape:",X_train.shape)
print("Test shape:",X_test.shape)
#==================================================================================
#BASELINE MODEL
baseline_pred = X_test['lag_1']
baseline_mae = mean_absolute_error(y_test,baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test,baseline_pred))

print("Baseline Mean Absolute Error",baseline_mae)#how much prediction is wrong
print("Root Mean Squared Error",baseline_rmse)
#=================================================================================
"""pipeline and random forest regression"""
#=================================================================================
num_features = [
    'lag_1','lag_6','lag_12','rolling_mean_6','rolling_std_6'


]

time_features =['hour','day']

preprocessor = ColumnTransformer(
    transformers=[
    ('num',StandardScaler(),num_features),
               ('time','passthrough',time_features)
    ]
)

random_forest_pipeline = Pipeline(steps=[
    ("preprocess",preprocessor),
    ('model',RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ))


])

random_forest_pipeline.fit(X_train,y_train)

prediction_random_forest = random_forest_pipeline.predict(X_test)

random_forest_mae = mean_absolute_error(y_test,prediction_random_forest)
random_forest_rmse = np.sqrt(mean_squared_error(y_test,prediction_random_forest))

print("RF Pipeline MAE : ", random_forest_mae)
print("RF Pipeline RMSE : ",random_forest_rmse)
'''My Random Forest model reduced MAE from 0.94 to 0.75 and RMSE from 1.86
to 1.54, achieving persistence baseline.'''
#==============================================================================
#Plot prediction
plt.figure(figsize=(12,4))
plt.plot(y_test.values[:300],label="Actual")
plt.plot(prediction_random_forest[:300],label="predicted")
plt.legend()
plt.title("Traffic Speed Prediction - Random Forest")
plt.xlabel("Hours")
plt.ylabel("speed")
plt.show()
#===================================================================================
'''Hyper Parameter tuning'''
#================================================================================
param_grid = {
    'model__n_estimators': [100,200],
    'model__max_depth': [None,10,20],
    'model__min_samples_split':[2,5],
    'model__min_samples_leaf':[1,2]
}

tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    random_forest_pipeline,
    param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train,y_train)

print("Best_Params:",grid_search.best_params_)
print("Best CV MAE :",-grid_search.best_score_)

best_model = grid_search.best_estimator_

prediction_tuned = best_model.predict(X_test)

tuned_mae = mean_absolute_error(y_test,prediction_tuned)
tuned_rmse = np.sqrt(mean_squared_error(y_test,prediction_tuned))

print("Tuned MAE :", tuned_mae)
print("Tuned RMSE:", tuned_rmse)
'''i tuned a Random Forest using TimeSeriesSplit and GridSearchCV,achiving
a reduction in MAE from 0.94(baseline) to 0.74, representing over 20% improvement,\
with controlled tree depth and leaf size to balance bias and variance'''

#______________________________________________________________________________
#______________________________________________________________________________
#CREATE JAM LABELS

def traffic_label(speed):
    if speed < 30:
        return 0 #jam
    elif speed < 45:
        return 1 #Slow
    else:
        return 2 #Free Flow
data['traffic_class'] = data['y'].apply(traffic_label)

print(data['traffic_class'].value_counts())
print(data.columns)
'''I discretized countinuous traffic speedninto categorical traffic states(jam,slow,free-flow)'''

'''Prepare Classification '''
X_cls = data.drop(['y','traffic_class'],axis=1)
y_cls = data["traffic_class"]
print(X_cls.columns)


#Time based split
X_train_c = X_cls.iloc[:split_idx]
X_test_c = X_cls.iloc[split_idx:]

y_train_c = y_cls.iloc[:split_idx]
y_test_c = y_cls.iloc[split_idx:]

print("Classification Train:",X_train_c.shape)
print("Classification  Test:",X_test_c.shape)

#Bulid Classification Pipeline
clf_pipeline = Pipeline(steps=[
     ('preprocess',preprocessor),
     ('model',RandomForestClassifier(
         n_estimators=150,
         random_state=42,
         n_jobs=-1,
         class_weight='balanced' #imbalance handling
     ))
 ])

#Hyper parameter tuning
param_grid_cls = {
    ''
    "model__n_estimators":[150,300],
    "model__max_depth":[None,10,20],
    "model__min_samples_split":[2,5],
    "model__min_samples_leaf":[1,2],
    "model__max_features":['sqrt','log2']
}

#Time-Series Cross Validation

tscv_cls = TimeSeriesSplit(n_splits=3)

grid_search_cls = GridSearchCV(
    clf_pipeline,
    param_grid_cls,
    cv=tscv_cls,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)

print("\nStating Jam Classifier HyperParameter Tuning...")
grid_search_cls.fit(X_train_c,y_train_c)

#Best Tuned Jam Model

best_jam_model = grid_search_cls.best_estimator_

print("\nBest Jam Model Parameters:")
print(grid_search_cls.best_params_)

print("\nBest CV F1(macro):",grid_search_cls.best_score_)

#Evaluate Tuned Model

pred_cls_tuned = best_jam_model.predict(X_test_c)

print("\n=== TUNED JAM PREDICTION RESULTS===")
print("Accuracy:",accuracy_score(y_test_c,pred_cls_tuned))

print("\nClassification Report:")
print(classification_report(
    y_test_c,pred_cls_tuned,
    target_names=["Jam","Slow","Free"]
))





ConfusionMatrixDisplay.from_estimator(
    best_jam_model,
    X_test_c,y_test_c,
    display_labels=['Jam','Slow','Free']
)
plt.title("Traffic Condition Classificaton Confusion Matrix")
plt.show()
#================================================================================
#Save Final Model
#===============================================================================
joblib.dump(best_model,"best_rf_regression_pipeline.pkl")
joblib.dump(best_jam_model,"best_jam_classifier_pipeline.pkl")

print("Both Models Saved Successfully <<^>>")


#________________________________________________________________________________
#________________________________________________________________________________
#UNSUPERVISED LEARNING :K-MEANS TRAFFIC PATTERN DISCOVERY
print("\nStrating Unsupervised Traffic Pattern Discovery(K-Means)...")

#select meaningful features for clustering
cluster_features = data[[
    "y",#current speed
    "lag_1",#recent speed
    "rolling_mean_6",#short -term trend
    "rolling_std_6",#variablity
    "hour"#time context
]]

print("Clustering feature shape:",
      cluster_features.shape)

#Scale features
scaler_unsup = StandardScaler()
X_cluster_scaled = scaler_unsup.fit_transform(cluster_features)

#Apply k-means
kmeans = KMeans(n_clusters=4,random_state=42,n_init=10)
clusters = kmeans.fit_predict(X_cluster_scaled)

#Add cluster labels to data
data['traffic_cluster'] = clusters

print("\nTraffic Cluster Distribution:")
print(data["traffic_cluster"].value_counts())

cluster_centers = pd.DataFrame(
    scaler_unsup.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_features.columns)

print("\nCluster Centers (Original Scale):")
print(cluster_centers)

cluster_name_map = {
    0: "Free Flow",
    1: "Moderate Traffic",
    2: "Heavy Congestion",
    3: "Stop-and-Go"
}

data['traffic_cluster_name'] = data['traffic_cluster'].map(cluster_name_map)
print("\nSample Cluster Naming:")
print(data[['traffic_cluster', 'traffic_cluster_name']].head())

print("\nTraffic Cluster Name Distribution:")
print(data['traffic_cluster_name'].value_counts())



cluster_hour = data.groupby(['hour', 'traffic_cluster']).size().unstack(fill_value=0)

print("cluster_hour shape:", cluster_hour.shape)
print(cluster_hour.head())


plt.figure(figsize=(10,5))

for cluster in cluster_hour.columns:
    name = cluster_name_map.get(cluster, f"Cluster {cluster}")
    plt.plot(cluster_hour.index,
             cluster_hour[cluster],
             label=name)

plt.title("Discovered Traffic Patterns by Hour (K-Means)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Observations")
plt.legend(title="Traffic Pattern")
plt.grid(True)
plt.show()

print("K-Means Unsupervised Analysis Completed")