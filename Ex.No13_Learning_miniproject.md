# Ex.No: 13 Mini Project
### DATE:                                                                  
### REGISTER NUMBER : 212222040180

### AIM:
To build a machine learning model that accurately predicts a car's fuel consumption (MPG) based on its technical specifications.

### Algorithm:
1.Data Collection and Loading: Load the dataset containing car specifications and fuel consumption data (MPG).

2.Data Preprocessing: Clean the data by handling missing values, converting data types, and normalizing features as needed.

3.Data Splitting: Divide the dataset into training and testing sets to evaluate model performance.

4.Model Training: Train a regression model, such as Random Forest Regressor, on the training set to predict MPG.

5.Evaluation and Prediction: Evaluate the model’s performance on the test set using metrics like RMSE and R-squared, and use the trained model to predict MPG for 
  new inputs.



### Program:
```python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('/content/auto-mpg.csv')

df.head()

df = df.replace('?', np.nan)
df = df.dropna()

df['horsepower'] = df['horsepower'].astype(float)

X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

import pickle

# Assuming 'model' is the trained model
# Save the model to a file
with open('fuel_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'fuel_prediction_model.pkl'")

import pickle
import numpy as np

# Load the model from the pickle file
with open('fuel_prediction_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define your sample input
# Replace these values with the specific details of the car you want to predict for
# Sample input format: [cylinders, displacement, horsepower, weight, acceleration, model year, origin]
sample_input = np.array([[6, 400, 130, 2004, 12.0, 70, 1]])

# Use the loaded model to make a prediction
predicted_mpg = loaded_model.predict(sample_input)

print(f"Predicted MPG for the sample car: {predicted_mpg[0]:.2f}")

```


### Output:

![image](https://github.com/user-attachments/assets/b57adb3c-2e96-414e-9ee3-66e125391213)


### Result:

The model successfully predicts a car's fuel efficiency (MPG) with reasonable accuracy based on its specifications.

