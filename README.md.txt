
# Project by:  
**Siva. R**  
Second Year, Mechanical Engineering  
ARM College of Engineering & Technology  
Course: Data Analysis in Mechanical Engineering  
```

---

## Machine Condition Classification Using Random Forest

This project focuses on predicting the condition of a machine using a machine learning model, specifically a **Random Forest Classifier**. The goal is to use various sensor data such as temperature, vibration, oil quality, and RPM to determine whether the machine is running normally or has a fault.

This project was done as part of learning how data analysis and machine learning can be applied in mechanical engineering, especially in predictive maintenance.

---

### Setup Instructions

To get started, install the required Python packages by running the command below in your terminal:

```bash
pip install -r requirements.txt
```

These packages are necessary for running the model and processing the input data.

---

### Files Required

Before running the prediction script, make sure the following files are available in your project directory:

* **`random_forest_model.pkl`** – The saved Random Forest model that has already been trained.
* **`scaler.pkl`** – A `StandardScaler` object used to normalize the input data.
* **`selected_features.pkl`** – A list of feature names that were used to train the model.

These files are essential for making predictions correctly, as they help maintain consistency with the original training process.

---

### How the Prediction Process Works

Here is a simple explanation of how the prediction is made:

1. **Load the Model and Other Files**
   Load the trained model, scaler, and feature list using the `joblib` library.

2. **Prepare Your Input Data**
   Create a DataFrame containing only one row, with all the features in the correct order. The names of the features must exactly match those used during training.

3. **Scale the Input**
   Normalize your input data using the loaded `scaler`, so that the data matches the format the model expects.

4. **Make a Prediction**
   Use the model's `.predict()` method to get the predicted condition (normal or faulty), and `.predict_proba()` to get the confidence levels of the prediction.

---

### Prediction Script Template

Here's a sample script you can use for making predictions:

```python
import joblib
import pandas as pd

# Load model, scaler, and selected features
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input (replace with real-time values)
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange columns in the correct order
new_data = new_data[selected_features]

# Scale the data
scaled_data = scaler.transform(new_data)

# Predict machine condition
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Machine Condition:", prediction[0])
print("Confidence Scores:", prediction_proba[0])
```

---

### Important Points to Remember

* Make sure all the input features are named and ordered exactly as during model training.
* Input values should be within reasonable operating ranges. Unexpected values may lead to incorrect predictions.
* The column order must not be changed.

---

### Model Updates (Optional)

If you want to retrain or improve the model in the future:

* Use the same steps for preprocessing.
* Make sure the features and scaling method remain consistent.
* Save the updated model and files again using `joblib`.

---

### Real-World Applications

This type of predictive model can be useful in:

* Preventive maintenance in manufacturing industries
* Monitoring machines in real-time using sensors
* Reducing downtime by identifying faults early

