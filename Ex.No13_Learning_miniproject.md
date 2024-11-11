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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Assuming you have already split the data into training and test sets,
# and have trained your model (model) and made predictions (y_pred) as shown earlier:

# y_test contains the actual MPG values (true values) from the test set
# y_pred contains the predicted MPG values from the model

# Example: 
# y_test = actual MPG values from your test data
# y_pred = predicted MPG values from your model

# Use this code after you have made predictions with the model:
y_pred = model.predict(X_test)  # Model predictions on the test set

# Evaluate the model using the actual test values (y_test) and predicted values (y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")


```


### Output:

![image](https://github.com/user-attachments/assets/b57adb3c-2e96-414e-9ee3-66e125391213)

```
 Accuracy
```

![image](https://github.com/user-attachments/assets/4328dd2a-c30b-441c-987e-9557274897d9)

### Result:

The model successfully predicts a car's fuel efficiency (MPG) with reasonable accuracy based on its specifications.

