import pandas as pd
import pickle
import streamlit as st
import os
from preprocessing import preprocess_pipeline

import warnings

warnings.filterwarnings('ignore')

# get project directory path
project_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

model_names = ['Random Forest', 'XGBoost']

# Define mapping dictionary
mapping = {
    0: 'Low',
    1: 'Intermediate',
    2: 'High'}

def run_model(model_name, data):
    if model_name == 'Random Forest':
        model = pickle.load(open(project_dir + 'models/rf.pkl', 'rb'))
    elif model_name == 'XGBoost':
        model = pickle.load(open(project_dir + 'models/xgb.pkl', 'rb'))
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    # Predict the class and the probability
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    probability = probability[0][prediction[0]]

    # Map the results to their respective classes
    return mapping[prediction[0]], probability

def main():
    # Set the width of the text input widget using CSS styling

    st.title('Games Rating Prediction')

    # layout
    # Get user input

    left_column1, right_column1 = st.columns(2)

    with left_column1:
        URL = st.text_input("Enter URL: ")
        ID = st.number_input("Enter ID: ")
        Name = st.text_input("Enter Name: ")
        Subtitle = st.text_input("Enter Subtitle: ")
        Icon_URL = st.text_input("Enter Icon_URL: ")
        User_Rating_Count = st.number_input("Enter User_Rating_Count: ")
        Price = st.number_input("Enter Price: ")
        In_app_Purchases = st.text_input("Enter In_app_Purchases: ")
        Developer = st.text_input("Enter Developer: ")
        Age_Rating = st.text_input("Enter Age_Rating: ")
        Languages = st.text_input("Enter Languages: ")

    with right_column1:
        Size = st.number_input("Enter Size: ")
        Primary_Genre = st.text_input("Enter Primary_Genre: ")
        Genres = st.text_input("Enter Genres: ")
        Release_Date = st.date_input("Enter Release_Date: ")
        Current_Version_Release_Date = st.date_input("Enter Current_Version_Release_Date: ")
        Description = st.text_area("Enter Description: ", value='')
        models_box = st.selectbox('Select a model to train', model_names)
        button = st.button('Submit')
        pred_label = st.write('Prediction: ')
        prediction = st.empty()
        prob_label = st.write('Probability: ')
        probability = st.empty()


    # Create button to trigger prediction
    if button:
        # Combine user input with dataset
        user_data = [[URL, ID, Name, Subtitle, Icon_URL, User_Rating_Count, Price, In_app_Purchases, Description,
                      Developer, Age_Rating, Languages, Size, Primary_Genre, Genres, Release_Date,
                      Current_Version_Release_Date]]

        df = pd.DataFrame(columns=['URL', 'ID', 'Name', 'Subtitle', 'Icon URL',
                                   'User Rating Count', 'Price', 'In-app Purchases', 'Description',
                                   'Developer', 'Age Rating', 'Languages',
                                   'Size', 'Primary Genre', 'Genres',
                                   'Original Release Date', 'Current Version Release Date',
                                   ])
        # append the user data as a new row to the DataFrame
        df = df.append(pd.DataFrame(user_data, columns=df.columns), ignore_index=True)

        df = preprocess_pipeline(df)

        scaler = pickle.load(open(project_dir + 'scalers/minmax_scaler.pkl', 'rb'))

        # df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

        # Get the feature names in the order they appear in the input DataFrame
        feature_names = df.columns

        # Transform the data using the scaler object
        df_scaled = scaler.transform(df)

        # Create a new DataFrame with the scaled data and the original feature names
        df_scaled = pd.DataFrame(df_scaled, columns=feature_names)

        selector = pickle.load(open(project_dir + 'encoders/selector.pkl', 'rb'))
        df_selected = selector.transform(df_scaled)

        # Predict results
        result, prob = run_model(models_box, df_selected)

        # make probability percentage to 2 decimal places
        prob = round(prob * 100, 2)

        # Print results
        prediction.write(result)
        probability.write(str(prob))


if __name__ == '__main__':
    main()
