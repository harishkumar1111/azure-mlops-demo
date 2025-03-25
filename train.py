import pandas as pd
from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace, Dataset
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics


# Authentication needs to be appiled while running the code the 1st time.
# ia = InteractiveLoginAuthentication(tenant_id='') #Add yours
# ws = Workspace(subscription_id= "", #Add yours
#     resource_group="", #Add yours
#     workspace_name= "", auth=ia) #Add yours
# print(f'worspace details {ws}')
run = Run.get_context()
df=pd.read_csv('car data.csv')
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset['Current Year']=2020
final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset=final_dataset.drop(['Current Year'],axis=1)
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestRegressor(n_estimators=100, random_state=0)
model = rf.fit(X_train,y_train)
predictions=model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, predictions)
run.log('MAE',np.float(mae))
mse =  metrics.mean_squared_error(y_test, predictions)
run.log('MSE',np.float(mse))
rmse =  np.sqrt(metrics.mean_squared_error(y_test, predictions))
run.log('RMSE',np.float(rmse))
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/random_forest_regression_model.pkl')
run.complete()