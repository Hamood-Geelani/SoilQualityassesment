import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data
train_data = pd.read_csv('/kaggle/input/spark4ai-prediction/soil_dataset/train.csv')
test_data = pd.read_csv('/kaggle/input/spark4ai-prediction/soil_dataset/test.csv')

# Data Preprocessing
# Handle missing values
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)
train_data['sensor1_mean'] = train_data.iloc[:, 7:23].mean(axis=1)
train_data['sensor1_std'] = train_data.iloc[:, 7:23].std(axis=1)
train_data['sensor1_min'] = train_data.iloc[:, 7:23].min(axis=1)
train_data['sensor1_max'] = train_data.iloc[:, 7:23].max(axis=1)

# Feature Engineering for sensor-2 data
train_data['sensor2_mean'] = train_data.iloc[:, 23:].mean(axis=1)
train_data['sensor2_std'] = train_data.iloc[:, 23:].std(axis=1)
train_data['sensor2_min'] = train_data.iloc[:, 23:].min(axis=1)
train_data['sensor2_max'] = train_data.iloc[:, 23:].max(axis=1)

# Normalize spectral reflectance data from sensor-2
scaler = StandardScaler()
train_data.iloc[:, 7:] = scaler.fit_transform(train_data.iloc[:, 7:])
X = train_data.drop(['Id', 'Property_A', 'Property_B', 'Property_C', 'Property_D', 'Property_E', 'Property_F'], axis=1)
y = train_data[['Property_A', 'Property_B', 'Property_C', 'Property_D', 'Property_E', 'Property_F']]
y['Predict']=y.iloc[:,:6].sum(axis=1)
y=y.drop(y.columns[:6],axis=1)
# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Concatenate sensor 1 and sensor 2 data
X_train_sensor_concat = pd.concat([X_train.iloc[:, 7:23], X_train.iloc[:, 23:]], axis=1)
X_val_sensor_concat = pd.concat([X_val.iloc[:, 7:23], X_val.iloc[:, 23:]], axis=1)

# Model Selection and Training - Random Forest
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_val)

# Evaluate the model
rmse_rf = mean_squared_error(y_val, y_pred_rf, squared=False)
print(f'RMSE on validation set (Random Forest): {rmse_rf}')

# Feature Engineering for sensor-1 data in test set
test_data['sensor1_mean'] = test_data.iloc[:, 7:17].mean(axis=1)
test_data['sensor1_std'] = test_data.iloc[:, 7:17].std(axis=1)
test_data['sensor1_min'] = test_data.iloc[:, 7:17].min(axis=1)
test_data['sensor1_max'] = test_data.iloc[:, 7:17].max(axis=1)

# Feature Engineering for sensor-2 data in test set
test_data['sensor2_mean'] = test_data.iloc[:, 23:].mean(axis=1)
test_data['sensor2_std'] = test_data.iloc[:, 23:].std(axis=1)
test_data['sensor2_min'] = test_data.iloc[:, 23:].min(axis=1)
test_data['sensor2_max'] = test_data.iloc[:, 23:].max(axis=1)

ids=test_data['Id']
test_features=test_data.drop("Id",axis=1)
test_features_scaled=scaler.transform(test_features)
test_features_scaled_df=pd.DataFrame(test_features_scaled,columns=test_features.columns)
# test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, 1:])

# Concatenate sensor 1 and sensor 2 data for test set
X_test_sensor_concat = pd.concat([test_data.iloc[:, 7:23], test_data.iloc[:, 23:]], axis=1)

# Test Data Prediction
test_predictions_rf = model_rf.predict(test_features_scaled_df)

# Create Submission File
submission_rf = pd.DataFrame({
    'Id': ids,
    'Predicted': test_predictions_rf  # Sum of all 6 predictions
})
submission_rf.to_csv('/kaggle/working/submission_rf.csv',index=False)