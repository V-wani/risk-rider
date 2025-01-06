import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np


#logo 

logo = r'''

#########################################################################################################
[     _____  ______ ____  _    _  _____        _____  _____            _____  ____  _   _  _____        ]
[     |  __ \|  ____|  _ \| |  | |/ ____|      |  __ \|  __ \     /\   / ____|/ __ \| \ | |/ ____|      ]
[     | |  | | |__  | |_) | |  | | |  __ ______| |  | | |__) |   /  \ | |  __| |  | |  \| | (___        ]
[     | |  | |  __| |  _ <| |  | | | |_ |______| |  | |  _  /   / /\ \| | |_ | |  | | . ` |\___ \       ]
[     | |__| | |____| |_) | |__| | |__| |      | |__| | | \ \  / ____ \ |__| | |__| | |\  |____) |      ]
[     |_____/|______|____/ \____/ \_____|      |_____/|_|  \_\/_/    \_\_____|\____/|_| \_|_____/       ]
[                                                                                                       ]
+=======================================================================================================+
    [                                                                                               ]
    [              "Hello, I am RiskRadar , your intelligent road safety assistant."                ]
    [    I analyze crash scenarios, predict severity, and provide actionable insights to            ]
    [                            make every journey safer.                                          ]
    [                                                                                               ]
    [   version V1.0                                                                                ]
    [   owned by DEBUG DRAGONS                                                                      ]
    [   github id:-                                                                                 ]
    [   youtube channel :-                                                                          ]
    [                                                                                               ]
    #################################################################################################'''


# dataset
file_path = r"Data Sheet - Sheet1.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
label_encoders = {}
for col in ['Gender', 'Vehicle_Type', 'Road_Type', 'Alcohol_Consumption', 'Crash_Type', 'Seatbelt_Usage', 'Road_Surface_Condition']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# target variable
target_encoder = LabelEncoder()
data['Crash_Severity'] = target_encoder.fit_transform(data['Crash_Severity'])
X = data.drop(columns=['Crash_Severity'])
y = data['Crash_Severity']

# filter variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("Classification Report:\n", logo , classification_report(y_test, y_pred))

# user input function
def predict_crash_severity():
    print("\n--- Predict Crash Severity ---")
    user_input = {}

    # speed checker
    while True:
        try:
            speed =float(input("Enter Vehicle Speed (e.g., up to 180 Km/hrs): "))
            if 0 < speed <= 150:
                user_input['Vehicle_Speed'] = speed
                break
            else:
                print("Invalid speed.\nPlease enter a speed between 0 to 150.")
        except ValueError:
            print("Invalid input.\nPlease enter a numeric value.")

    # crashtime checker
    while True:
        try:
            crash_time = float(input("Enter Crash Time (in hrs e.g., 2): "))
            if 0 < crash_time <= 24:
                user_input['Crash_Time'] = crash_time
                break
            else:
                print("Invalid input.\nPlease enter a value between 0 and 24.")
        except ValueError:
            print("Invalid input.\nPlease enter a numeric value.")

    # age checker
    while True:
        try:
            age = int(input("Enter Driver's Age (e.g., 30): "))
            if 18 <= age <= 85:
                user_input['Age'] = age
                break
            else:
                print("Invalid age.\nDriver age must be between 18 and 85 years old.")
        except ValueError:
            print("Invalid input.\nPlease enter an integer value.")

    # gender checker
    while True:
        gender = input("Enter Gender (for male - M or female - F): ").lower()
        if gender in ['m', 'f']:
            gender = "Male" if gender == 'm' else "Female"
            user_input['Gender'] = label_encoders['Gender'].transform([gender])[0]
            break
        print("Invalid input.\nPlease enter 'M' or 'F'.")

    # vehical checker
    while True:
        vehicle = input("Enter Vehicle Type (Two wheeler - T /Car - C /Heavy Vehicle - H): ").lower()
        if vehicle in ['t', 'c', 'h']:
            vehicle = "T.W" if vehicle == 't' else "Car" if vehicle == 'c' else "Heavy Vehicle"
            user_input['Vehicle_Type'] = label_encoders['Vehicle_Type'].transform([vehicle])[0]
            break
        print("Invalid input.\nPlease enter 'T', 'C', or 'H'.")

    # lane checker
    while True:
        try:
            lane = int(input("Enter Number of Lanes (1 - 3, e.g., 2): "))
            if 1 <= lane <= 3:
                user_input['Number_of_Lanes'] = lane
                break
            else:
                print("Invalid input.\nNumber of lanes must be between 1 and 3.")
        except ValueError:
            print("Invalid input.\nPlease enter an integer value.")

    # lane width checker
    while True:
        try:
            lane_width = float(input("Enter Lane Width (3 - 4, e.g., 3.565 m): "))
            if 3 <= lane_width <= 4:
                user_input['Lane_Width'] = lane_width
                break
            else:
                print("Invalid lane width.\nPlease enter a value between 3 and 4.")
        except ValueError:
            print("Invalid input.\nPlease enter a numeric value.")

    # road type checker
    while True:
        road_type = input("Enter Road Type (Urban - U /Rural - R): ").lower()
        if road_type in ['u', 'r']:
            road_type = "Urban" if road_type == 'u' else "Rural"
            user_input['Road_Type'] = label_encoders['Road_Type'].transform([road_type])[0]
            break
        print("Invalid input.\nPlease enter 'U' or 'R'.")

    # alcohol consumption checker
    while True:
        alcohol = input("Alcohol Consumption (Yes - Y / No - N): ").lower()
        if alcohol in ['y', 'n']:
            alcohol = "Yes" if alcohol == 'y' else "No"
            user_input['Alcohol_Consumption'] = label_encoders['Alcohol_Consumption'].transform([alcohol])[0]
            break
        print("Invalid input.\nPlease enter 'Y' or 'N'.")

    # crash type checker
    while True:
        crash = input("Enter Crash Type (Head-on - 1 / Rear-end - 2): ")
        if crash in ['1', '2']:
            crash = "Head-on" if crash == '1' else "Rear-end"
            user_input['Crash_Type'] = label_encoders['Crash_Type'].transform([crash])[0]
            break
        print("Invalid input.\nPlease enter '1' or '2'.")

    # sefety gadgets checker
    while True:
        seatbelt = input("Seatbelt or Helmet Usage (Yes - Y / No - N): ").lower()
        if seatbelt in ['y', 'n']:
            seatbelt = "Yes" if seatbelt == 'y' else "No"
            user_input['Seatbelt_Usage'] = label_encoders['Seatbelt_Usage'].transform([seatbelt])[0]
            break
        print("Invalid input.\nPlease enter 'Y' or 'N'.")

    # speed limit checker
    while True:
        try:
            speed_limit = float(input("Enter Speed Limit (e.g., 60): "))
            if 30 <= speed_limit <= 120:
                user_input['Speed_Limit'] = speed_limit
                break
            else:
                print("Invalid speed limit.\nPlease enter a value between 30 to 120.")
        except ValueError:
            print("Invalid input.\nPlease enter a numeric value.")

    # surface checker
    while True:
        surface = input("Enter Road Surface Condition (Dry - D / Wet - W / Icy - I): ").lower()
        if surface in ['d', 'w', 'i']:
            surface = "Dry" if surface == 'd' else "Wet" if surface == 'w' else "Icy"
            user_input['Road_Surface_Condition'] = label_encoders['Road_Surface_Condition'].transform([surface])[0]
            break
        print("Invalid input.\nPlease enter 'd', 'w', or 'i'.")

    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_scaled)
    severity = target_encoder.inverse_transform(prediction)

    print(f"Predicted Crash Severity: {severity[0]}")

predict_crash_severity()
