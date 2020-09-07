# Predict Acid Rain Based On Atmospheric Pollutant


### Import libraries


```python
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import datetime

from sklearn import preprocessing 

#*************************knn*************************************
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

#*************************Naive Bayes *********************
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

#*************************Naive Bayes stratified approach*********************
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

```

### Extract rain events
The dataset consists of hourly data of amount of rainfall and pH level of the rainfall of 25 stations.

However, not all stations have this data. In the stations that contain this information, not all data is valid.
Hence those events are extracted where there is a valid rainfall event (defined by threshold of rainfall > 0.05mm) and there is a valid pH information of the particular rainfall.


```python
def filter_rain_events(raw_station_data, RAIN_THRESHOLD=0.05 ):
	count_rain_data_points = 0

	rain_events_only = raw_station_data.copy()
	for each_station in rain_events_only:

		# remove NaN rows
		station_cleaned = rain_events_only[each_station].dropna()
        
		# remove events with no rainfall (NR)
		station_cleaned = station_cleaned[station_cleaned.RAINFALL != 'NR']
		station_cleaned = station_cleaned[station_cleaned.PH_RAIN != 'NR']
		station_cleaned = station_cleaned[station_cleaned.PH_RAIN != 'nan']

		# filter events with little rain
		station_cleaned = station_cleaned[station_cleaned.RAINFALL.astype(float) >= RAIN_THRESHOLD]

       #Rain event of each station and the total rain event count over Taiwan is extracted.
		rain_events_only[each_station] = station_cleaned
		count_rain_data_points = count_rain_data_points+len(station_cleaned)

	return rain_events_only, count_rain_data_points
```

## Acid rain

If the pH level of a liquid is below 7, it is termed as acidic. The pH level of the rainwater is typically acidic (around 5.6) due to the natural gases like C02,gasses from volcano etc. However if the pH level drops below 5, it is harmful. 

After extracting the rain event which has valid pH values, we classify the event as acidic rain or non-acidic.



```python
def mark_acid_rain(stations, PH_ACID_RAIN_THRESHOLD=5):
	count_acid_rain_data_points = 0
    
   
	for each_station in stations:
       #Initialize all the events as non-acidic
		stations[each_station]['acid_rain'] = False

		for index, row in stations[each_station].iterrows():
			if float(stations[each_station].loc[index,'PH_RAIN']) <= PH_ACID_RAIN_THRESHOLD:
				stations[each_station].loc[index,'acid_rain'] = True 
              # Get the count of total acid rain events.
				count_acid_rain_data_points = count_acid_rain_data_points+1
    
	return stations, count_acid_rain_data_points
```

### Window

The raw data is split into 1 hour long events.In the Windowing process,  consecutive rain data points are accumulated into Rain
Windows.  A Rain Window is considered as acidic when at least one of it’s data points is labeled as acidic.


```python

def build_rain_windows(rain_events, window_mode=True):

	rain_windows = [] # list of rain windows, of all stations
	count_windows=0


	for each in rain_events:
		# take last data point -> new rain event
		rain_window = pd.DataFrame(columns=rain_events[each].columns)
		# iterrate background through rain events, until difference is > 1h

		if window_mode:
			for index, row in rain_events[each].iterrows(): 
				row.name = index # make sure index stays the same
				rain_window = rain_window.append(row)

				# HINT: the index is used to check if the next data point was also rain
                #(this only works since the data points are sorted by time)
				if index+1 in rain_events[each].index:
					continue
				else:
					rain_windows.append(rain_window)
					rain_window = pd.DataFrame(columns=rain_events[each].columns)
					count_windows=count_windows+1
		else:
			for index, row in rain_events[each].iterrows(): 
				row.name = index # make sure index stays the same
				rain_window = rain_window.append(row)
				rain_windows.append(rain_window)
				rain_window = pd.DataFrame(columns=rain_events[each].columns)
				count_windows=count_windows+1
			# if previous data points received, append window tripple (pre_values, window, acid)


	return rain_windows, count_windows
```

### Extract previous data points

The rain water becomes acidic if the amount of pollutant in the atmosphere (Nox and SO2) is high. Hence we need to extract the pollutant level before the event of the rainfall. 

It is checked if there is a valid data for the NOx and SO2 one hour before every rain event.



```python
def get_previous_data_points(raw_data, rain_windows, number_of_previous_data_points=10):
    
	analytical_data = []
	count_acidic_windows=0
	count_no_valid_previous_data=0
	acid_rain_window = False
    
	# rermark the dataset is pre cleaned, if data was NA it will be ignored
	for each in rain_windows:

		index=each.index.min()
       # remark only one station can be in each window
		station=each.station.min() 
        
       #Extract the data of  previous hours of data.
		previous_data_points = raw_data[station].loc[index-number_of_previous_data_points:index-1]


       #Extract information if there was acid rain in the previous hours.
		if True in each["acid_rain"].unique():
			acid_rain_window = True
			count_acidic_windows=count_acidic_windows+1
		else:
			acid_rain_window = False

		if len(previous_data_points) > 0:
           # only consider data when you have the NOx and SO2 1h before rain
			if previous_data_points.loc[index-1]["NOx"] != "NR" and previous_data_points.loc[index-1]["SO2"] != "NR":
				analytical_data.append([previous_data_points, each, acid_rain_window])
		else: 
			count_no_valid_previous_data=count_no_valid_previous_data+1
            
	return analytical_data, count_acidic_windows, count_no_valid_previous_data

```

### Additional KPIs and handling time series data.

Now we have extracted rain event, ensured that there is Nox and So2 data one hour prior to the event. 

In case of rainfall, the pollutants dissolve in the rainwater and causes acid rain. Hence the rainfall event effects the acidity of the rainfall event in the future. Hence rainfall is considered for the prediction.

Observing the raw dataset, we see that rainwater is acidic during several months. These months have a higher temperature than the reset. Hence ambient temperature is considered for the prediction.

In the previous function, we extract data over last several hours of the rain event. An average/Sum/Median of the pollutants in last several hours can provide a better estimate if the rainfall is acidic or not. 


To determine 
    a) If inclusion of rainfall improves the performance
    
    b) If inclusion of ambient temperature improves the performance
    
    c) If Average of last several hours of data of KPIs provide better performance
    
    d) If Sum of last several hours of data of KPIs provide better performance
    
    e) If Median of last several hours of data of KPIs provide better performance
    
    
the function is written to provide all different combination of results. Later the performance can be compared to determine the best performing model.




```python
def setup_matrix(analytical_data, function="avg", include_rain=True, include_amb_temp=True, normalisation=True):
	count_acidic_windows = 0
	count_non_acidic_windows = 0

	analytical_matrix_results = pd.DataFrame(columns=["ACID_RAIN"]) 
    
    
   #To determine performance for the different combinations of KPI
	if include_rain and include_amb_temp:
		analytical_matrix = pd.DataFrame(columns=["NOx","SO2","AMB_TEMP","RAINFALL"]) 
	elif include_rain and not include_amb_temp:
		analytical_matrix = pd.DataFrame(columns=["NOx","SO2","RAINFALL"]) 
	elif not include_rain and include_amb_temp:
		analytical_matrix = pd.DataFrame(columns=["NOx","SO2","AMB_TEMP"]) 
	else:
		analytical_matrix = pd.DataFrame(columns=["NOx","SO2"]) 


   #calculate averge & sum & median as requested by the function for the KPIs over several hours.
	for each in analytical_data:
		SO2=0
		NOx=0
		AMB_TEMP=0
		RAINFALL=0

		each[0] = each[0].replace("NR", np.nan, regex=True)

		if function=="avg":
			SO2 = each[0]["SO2"].astype(float).mean()
			NOx = each[0]["NOx"].astype(float).mean()
			if include_rain:
				RAINFALL = each[0]["RAINFALL"].astype(float).mean()
			if include_amb_temp:
				AMB_TEMP = each[0]["AMB_TEMP"].astype(float).mean()
		elif function=="sum":
			SO2 = each[0]["SO2"].astype(float).sum()
			NOx = each[0]["NOx"].astype(float).sum()
			AMB_TEMP = each[0]["AMB_TEMP"].astype(float).sum()
			if include_rain:
				RAINFALL = each[0]["RAINFALL"].astype(float).sum()
			if include_amb_temp:
				AMB_TEMP = each[0]["AMB_TEMP"].astype(float).sum()
		elif function=="median":
			SO2 = each[0]["SO2"].astype(float).median()
			NOx = each[0]["NOx"].astype(float).median()
			AMB_TEMP = each[0]["AMB_TEMP"].astype(float).median()
			if include_rain:
				RAINFALL = each[0]["RAINFALL"].astype(float).median()
			if include_amb_temp:
				AMB_TEMP = each[0]["AMB_TEMP"].astype(float).median()
		
		# REMARK: for statistical purposes 
		if each[2] == True:
			count_acidic_windows=count_acidic_windows+1
		else:
			count_non_acidic_windows=count_non_acidic_windows+1


       #Output a matrix which contains the sum/average/median of NOx,SO2 and if requested Rainfall,Ambient temperature.
		if include_rain and include_amb_temp:
			appending_dict = {"NOx": NOx, "SO2": SO2, "RAINFALL": RAINFALL, "AMB_TEMP": AMB_TEMP }
		elif include_rain and not include_amb_temp:
			appending_dict = {"NOx": NOx, "SO2": SO2, "RAINFALL": RAINFALL }
		elif not include_rain and include_amb_temp:
			appending_dict = {"NOx": NOx, "SO2": SO2, "AMB_TEMP": AMB_TEMP }
		else:
			appending_dict = {"NOx": NOx, "SO2": SO2}


		analytical_matrix = analytical_matrix.append(appending_dict, ignore_index=True)
       
       #Provide the information about if the rainfall event was acidic or not
		analytical_matrix_results = analytical_matrix_results.append({"ACID_RAIN": int(each[2])}, ignore_index=True)

	return analytical_matrix, analytical_matrix_results, count_acidic_windows,count_non_acidic_windows
```

### Measuring performance.

The performance of the algorithm is measured via F1 score. However Precision score,Recall score and accuracy is also recorded.

Following classifiers are used :

1) knn classifier

2) Naive Bayes classifier

3) Naive Bayes classifier with 5 fold stratified approach

4) Naive Bayes classifier with 10 fold stratified approach


```python
def measure_results(analytical_dataframe, analytical_dataframe_results):
    
	test_results = pd.DataFrame(columns=["classifier", 
                                      "knn_parameters", 
                                      "test_size", "f1_score", 
                                      "precsission", "recall", "accuracy", 
                                      "confusion_matrix"])
    
	dataset_all_stations=analytical_dataframe.fillna(0.0)
	pH_level_all_stations = []
       
	for each in analytical_dataframe_results.fillna(0.0).values:
		pH_level_all_stations.append(each[0])
        
   #split the data to training and test set (Ratio 80% and 20%)
	data_train, data_test, target_train, target_test = train_test_split(dataset_all_stations,pH_level_all_stations,test_size=0.2, random_state=1)

   #Normalizing the data set
	x = data_train.fillna(0.0).values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	min_max_scaler.fit(x)
	x_scaled = min_max_scaler.transform(x)
	data_train = pd.DataFrame(x_scaled) # normalised data set

	x = data_test.fillna(0.0).values #returns a numpy array
   #scale the test set using the same normalizing factor as training set.
	x_scaled = min_max_scaler.transform(x)
	data_test = pd.DataFrame(x_scaled) # normalised data set
    
	dataset_all_stations_normalized = preprocessing.MinMaxScaler().fit_transform(dataset_all_stations)

   #***********************knn classifier*************************************
	knn_estimator = KNeighborsClassifier()
   #Set the parameters
	parameters = {
	 	'n_neighbors': range(2, 9), 
	 	'algorithm': ['ball_tree', 'kd_tree', 'brute']}
	stratified_10_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	grid_search_estimator = GridSearchCV(knn_estimator, parameters, scoring='f1_macro', cv=stratified_10_fold_cv)

   #Determine the best algorithm and optimal neighbour which gives the best performance.
	grid_search_estimator.fit(data_train,target_train) 
   #Perform the prediction on the test set with the same parameter.
	predict = grid_search_estimator.predict(data_test)

   #Output results
	test_results = test_results.append({"classifier": "knn", "knn_parameters": format(grid_search_estimator.best_params_), 
                                     "test_size": 0.2, 
                                     "f1_score": f1_score(target_test,predict, average="macro"), 
                                     "precsission": precision_score(target_test, predict, average="macro"),
                                     "recall": recall_score(target_test, predict, average="macro"), 
                                     "accuracy": accuracy_score(target_test, predict), 
                                     "confusion_matrix": confusion_matrix(target_test, predict)}, ignore_index=True)
    
    
   #**************************Naive Bayes*************************************
	naive_bayes = GaussianNB()
	naive_bayes.fit(data_train,target_train) 
	predict = naive_bayes.predict(data_test)
    
   #Output results
	test_results = test_results.append({"classifier": "nb", "knn_parameters": "", "test_size": 0.2,
                                     "f1_score": f1_score(target_test,predict, average="macro"),
                                     "precsission": precision_score(target_test, predict, average="macro"),
                                     "recall": recall_score(target_test, predict, average="macro"),
                                     "accuracy": accuracy_score(target_test, predict),
                                     "confusion_matrix": confusion_matrix(target_test, predict)}, ignore_index=True)

   #***********************Naive Bayes 5 fold stratified*********************
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	predict = cross_val_predict(naive_bayes, dataset_all_stations_normalized, pH_level_all_stations, cv=cv)

   #Output results
	test_results = test_results.append({"classifier": "nb5", "knn_parameters": "", 
                                     "test_size": "5 fold",
                                     "f1_score": f1_score(pH_level_all_stations,predict, average="macro"),
                                     "precsission": precision_score(pH_level_all_stations, predict, average="macro"), 
                                     "recall": recall_score(pH_level_all_stations, predict, average="macro"),
                                     "accuracy": accuracy_score(pH_level_all_stations, predict), 
                                     "confusion_matrix": confusion_matrix(pH_level_all_stations, predict)}, ignore_index=True)
    
    
   #***********************Naive Bayes 10 fold stratified*********************
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	predict = cross_val_predict(naive_bayes, dataset_all_stations_normalized, pH_level_all_stations, cv=cv)

   #Output results
	test_results = test_results.append({"classifier": "nb10", "knn_parameters": "", 
                                     "test_size": "10 fold", 
                                     "f1_score": f1_score(pH_level_all_stations,predict, average="macro"), 
                                     "precsission": precision_score(pH_level_all_stations, predict, average="macro"),
                                     "recall": recall_score(pH_level_all_stations, predict, average="macro"), 
                                     "accuracy": accuracy_score(pH_level_all_stations, predict),
                                     "confusion_matrix": confusion_matrix(pH_level_all_stations, predict)}, ignore_index=True)
    
   #Return the results of all the classifiers. 
	return test_results

```

### Analyze results

Given previous data of the rain events, this function combines the previous data using sum/average/median of KPIs (setup_matrix)and provides the result of the classifier (measure_results).



```python

def analyse_results(analytical_windows, function="avg", include_rain=True, include_amb_temp=True):
	print("...calculating results = include rain:", include_rain, ",include ambient temperature:", include_amb_temp, "and aggregation function:", function)
	analytical_dataframe, analytical_dataframe_results ,count_acidic_windows, count_non_acidic_windows = setup_matrix(analytical_windows, function=function, include_rain=include_rain, include_amb_temp=include_amb_temp)
	return measure_results(analytical_dataframe, analytical_dataframe_results)

```

### Read the data


```python
dataset = pd.read_csv('./2015_Air_quality_in_northern_Taiwan.csv', low_memory=False)
```

### Throwing out invalid data

The data set contains several invalid data

\# indicates invalid value by equipment inspection

\* indicates invalid value by program inspection

x indicates invalid value by human inspection

NR indicates no rainfall



```python
dataset = dataset.mask(dataset.applymap(lambda x: '#' in x if  isinstance(x,str) else False))  
dataset = dataset.mask(dataset.applymap(lambda x: '*' in x if  isinstance(x,str) else False)) 
dataset = dataset.mask(dataset.applymap(lambda x: 'x' in x if  isinstance(x,str) else False))  
```

### Running the algorithm for various combination

The below code is configured for the best parameter settings. In case different combination results are required,refer to the comment section in the code to obtain different configuration results.



```python
raw_data = dict(tuple(dataset[['time','station','NOx', 'SO2', 'PH_RAIN', 'RAINFALL', 'AMB_TEMP']].sort_values('time').groupby('station')))

RAIN_THRESHOLD = 0.01
ph_value=[4.5]
windows=[False]
data_points=[600]
include_rain=[True]
include_amb_temp=[True]
function=["avg"]
"""
Different possible combination of parameter setting

ph_value=[4.5,5.0]
windows=[False]
data_points=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,50,100,150,300,600]
include_rain=[False,True]
include_amb_temp=[False,True]
function=["avg", "sum", "median"]
"""

combined_results = pd.DataFrame(columns=["previous_datapoints", "nox", "sox", "rain", "ambient_temperature", "function", "windowed", "classifier", "knn_parameters", "ph_value", "test_size", "acid_rain_events", "non_acid_rain_events", "rain_TH", "f1_score", "precsission", "recall", "accuracy", "confusion_matrix"])

i=0
for each_ph_value in ph_value:
	rain_events, count_rain_events = filter_rain_events(raw_data, RAIN_THRESHOLD)
	print("FOUND", count_rain_events, "rain data points, with less than", RAIN_THRESHOLD, "L/hour rain, in all stations combined.")
	marked_rain_events, count_acid_rain_events = mark_acid_rain(rain_events, each_ph_value)
	print("FOUND", count_acid_rain_events, "acid rain data points, with PH_RAIN value below", each_ph_value, ", in all stations combined.")
	for each_window in windows:
		for each_data_point in data_points:
			for each_rain in include_rain:
				for each_amb_temp in include_amb_temp:
					if each_data_point > 1:
						for each_function in function:
							rain_windows, count_rain_windows, = build_rain_windows(marked_rain_events, window_mode=each_window)

							analytical_windows, count_acid_rain_windows, count_no_valid_previous_data = get_previous_data_points(raw_data, rain_windows, each_data_point) 
							test_results = analyse_results(analytical_windows, function=each_function, include_rain=each_rain, include_amb_temp=each_amb_temp)

							current_configuration = {"previous_datapoints": each_data_point, "nox" : True, "sox": True, "rain": each_rain, "ambient_temperature": each_amb_temp, "function": each_function, "windowed": each_window, "ph_value": each_ph_value, "test_size": 0.2, "acid_rain_events": count_rain_windows-count_acid_rain_windows, "non_acid_rain_events": count_acid_rain_windows, "rain_TH": RAIN_THRESHOLD} 

							for each in test_results.to_dict(orient='records'):
								each.update(current_configuration)
								combined_results = combined_results.append(each, ignore_index=True)




					else:
						rain_windows, count_rain_windows, = build_rain_windows(marked_rain_events, window_mode=each_window)
		
						analytical_windows, count_acid_rain_windows, count_no_valid_previous_data = get_previous_data_points(raw_data, rain_windows, each_data_point)
						test_results = analyse_results(analytical_windows, function="avg", include_rain=each_rain, include_amb_temp=each_amb_temp)

						current_configuration = {"previous_datapoints": each_data_point, "nox" : True, "sox": True, "rain": each_rain, "ambient_temperature": each_amb_temp, "function": "", "windowed": each_window, "ph_value": each_ph_value, "test_size": 0.2, "acid_rain_events": count_rain_windows-count_acid_rain_windows, "non_acid_rain_events": count_acid_rain_windows, "rain_TH": RAIN_THRESHOLD} 

						for each in test_results.to_dict(orient='records'):
							each.update(current_configuration)
							combined_results = combined_results.append(each, ignore_index=True)

						

						
					if i%50 == 0:
							print("run", i, combined_results)
					else: 
						print("run", i, "next intermediate print in" ,-(i % 50 - 50), "runs")

					i=i+1
```

    FOUND 3337 rain data points, with less than 0.01 L/hour rain, in all stations combined.
    FOUND 1740 acid rain data points, with PH_RAIN value below 4.5 , in all stations combined.
    ...calculating results = include rain: True ,include ambient temperature: True and aggregation function: avg
    run 0   previous_datapoints   nox   sox  rain ambient_temperature function windowed  \
    0                 600  True  True  True                True      avg    False   
    1                 600  True  True  True                True      avg    False   
    2                 600  True  True  True                True      avg    False   
    3                 600  True  True  True                True      avg    False   
    
      classifier                                knn_parameters  ph_value  \
    0        knn  {'algorithm': 'ball_tree', 'n_neighbors': 3}       4.5   
    1         nb                                                     4.5   
    2        nb5                                                     4.5   
    3       nb10                                                     4.5   
    
       test_size acid_rain_events non_acid_rain_events  rain_TH  f1_score  \
    0        0.2             1597                 1740     0.01  0.825520   
    1        0.2             1597                 1740     0.01  0.585623   
    2        0.2             1597                 1740     0.01  0.601680   
    3        0.2             1597                 1740     0.01  0.603717   
    
       precsission    recall  accuracy           confusion_matrix  
    0     0.828891  0.825091  0.826347     [[253, 72], [44, 299]]  
    1     0.594939  0.589693  0.592814   [[154, 171], [101, 242]]  
    2     0.611700  0.604999  0.610129  [[775, 822], [479, 1261]]  
    3     0.614003  0.607062  0.612227  [[777, 820], [474, 1266]]  
    


```python
combined_results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>previous_datapoints</th>
      <th>nox</th>
      <th>sox</th>
      <th>rain</th>
      <th>ambient_temperature</th>
      <th>function</th>
      <th>windowed</th>
      <th>classifier</th>
      <th>knn_parameters</th>
      <th>ph_value</th>
      <th>test_size</th>
      <th>acid_rain_events</th>
      <th>non_acid_rain_events</th>
      <th>rain_TH</th>
      <th>f1_score</th>
      <th>precsission</th>
      <th>recall</th>
      <th>accuracy</th>
      <th>confusion_matrix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>600</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>avg</td>
      <td>False</td>
      <td>knn</td>
      <td>{'algorithm': 'ball_tree', 'n_neighbors': 3}</td>
      <td>4.5</td>
      <td>0.2</td>
      <td>1597</td>
      <td>1740</td>
      <td>0.01</td>
      <td>0.825520</td>
      <td>0.828891</td>
      <td>0.825091</td>
      <td>0.826347</td>
      <td>[[253, 72], [44, 299]]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>600</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>avg</td>
      <td>False</td>
      <td>nb</td>
      <td></td>
      <td>4.5</td>
      <td>0.2</td>
      <td>1597</td>
      <td>1740</td>
      <td>0.01</td>
      <td>0.585623</td>
      <td>0.594939</td>
      <td>0.589693</td>
      <td>0.592814</td>
      <td>[[154, 171], [101, 242]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>600</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>avg</td>
      <td>False</td>
      <td>nb5</td>
      <td></td>
      <td>4.5</td>
      <td>0.2</td>
      <td>1597</td>
      <td>1740</td>
      <td>0.01</td>
      <td>0.601680</td>
      <td>0.611700</td>
      <td>0.604999</td>
      <td>0.610129</td>
      <td>[[775, 822], [479, 1261]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>600</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>avg</td>
      <td>False</td>
      <td>nb10</td>
      <td></td>
      <td>4.5</td>
      <td>0.2</td>
      <td>1597</td>
      <td>1740</td>
      <td>0.01</td>
      <td>0.603717</td>
      <td>0.614003</td>
      <td>0.607062</td>
      <td>0.612227</td>
      <td>[[777, 820], [474, 1266]]</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
