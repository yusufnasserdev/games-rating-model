from sklearn.metrics import accuracy_score
import pickle

import cv2
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from skimage.feature import local_binary_pattern

import nltk
import re

from datetime import datetime
from tqdm import tqdm

import requests
import os
import shutil

import warnings

stop_words = set(stopwords.words('english'))

warnings.filterwarnings('ignore')

# get previous directory path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

def download_image(url, filename):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(filename, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

def download_icons(df):
    # Convert to string
    df['Icon URL'] = df['Icon URL'].astype(str)

    # Create a folder to store the images
    if not os.path.exists(f"{project_dir}icons/deploy"):
        os.makedirs(f"{project_dir}icons/deploy")

    # Download the images
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not os.path.exists(f"{project_dir}icons/deploy/{i}.png"):
            download_image(row['Icon URL'], f"{project_dir}icons/deploy/{i}.png")

    # Replace the URL with the icon filename which is the index of the row
    df['Icon URL'] = df.apply(
        lambda row: f"{project_dir}icons/deploy/{row.name}.png", axis=1)

def web_scrapping(_df):
    data = pd.DataFrame(columns=["ID", "Reviews"])
    
    for url, id_value in zip(_df['URL'], _df['ID']):
        # Redirect to the reviews page
        url += "?see-all=reviews"
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code == 200:  
            soup = BeautifulSoup(response.text, 'html.parser')
            blocks = soup.findAll("blockquote")
            review_list = []
            for blockquote in blocks:
                review = blockquote.find('p').text
                review_list.append(review)
                
            if len(review_list) != 0:
                new_row = {'ID': id_value, "Reviews": review_list}
                data = data.append(new_row, ignore_index=True)
    
    data.to_csv(project_dir + 'datasets/deploy/reviews.csv', index=False)

def reviews_splitting(data):
    for i in range(len(data)):
        data.at[i, 'Reviews'] = data.at[i, "Reviews"].split("',")
        data.at[i, "ID"] = data.at[i, "ID"]

    data = data.explode('Reviews')
    return data

def reviews_cleaning(data):
    # Convert text to lowercase
    data['Reviews'] = data['Reviews'].apply(lambda x: str(x).lower())

    # Replace newline characters with an empty string
    data['Reviews'] = data['Reviews'].apply(lambda x: re.sub(r'\\n', ' ', x))

    # Remove black squares
    data['Reviews'] = data['Reviews'].apply(
        lambda x: re.sub(r'\\u25a0', '', x))

    # Remove special characters and punctuations
    data['Reviews'] = data['Reviews'].apply(
        lambda x: re.sub(r'[^\w\s]+', '', x))

    # Remove numbers
    data['Reviews'] = data['Reviews'].apply(
        lambda x: " ".join([word for word in x.split() if not any(char.isdigit() for char in word)]))

    # Remove extra whitespaces
    data['Reviews'] = data['Reviews'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # Remove stop words
    data['Reviews'] = data['Reviews'].apply(
        lambda x: " ".join([word for word in x.lower().split() if word not in stop_words]))

    # Remove empty strings
    data = data[data['Reviews'].apply(lambda x: len(x) > 0)]

    # Group by ID
    data = data.groupby('ID')['Reviews'].apply(list).reset_index()

    return data

def download_reviews(df):

    web_scrapping(df)

    df_reviews = pd.read_csv(project_dir + 'datasets/deploy/reviews.csv')

    df_reviews = reviews_splitting(df_reviews)

    df_reviews = reviews_cleaning(df_reviews)

    # Merge The Sentiment with the original dataset
    df = df.merge(df_reviews, on='ID', how='left')

    return df

def date_preprocess(_df):
    # Use dateparse to parse the dates
    _df['Original Release Date'] = pd.to_datetime(
        _df['Original Release Date'], format='%d/%m/%Y', errors='coerce')
    _df['Current Version Release Date'] = pd.to_datetime(
        _df['Current Version Release Date'], format='%d/%m/%Y', errors='coerce')

    # Convert the datetime to ordinal
    _df['Original Release Date'] = _df['Original Release Date'].apply(
        lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    _df['Current Version Release Date'] = _df['Current Version Release Date'].apply(
        lambda x: x.toordinal() if pd.notnull(x) else np.nan)

    # Impute missing values using simple imputer with median strategy

    simple_imputer = pickle.load(
        open(project_dir + 'imputers/classification/date_simple.pkl', 'rb'))

    _df[['Original Release Date', 'Current Version Release Date']] = simple_imputer.transform(
        _df[['Original Release Date', 'Current Version Release Date']])

    # Create a new column with the age of the game
    _df['game_age'] = datetime.now().toordinal() - _df['Original Release Date']

    # Create a new column with the time since the last update
    _df['last_update'] = datetime.now().toordinal() - \
        _df['Current Version Release Date']

    # Create a new column with the maintaning period
    _df['maintaning_period'] = _df['game_age'] - _df['last_update']

    return _df

def dev_preprocess_target_enc(_df):
    # Convert Developer column to string
    _df['Developer'] = _df['Developer'].astype(str)
    _df['Developer'] = _df['Developer'].str.replace("'", "").str.strip('[]')

    # Perform target encoding on Developer column

    te = pickle.load(open(project_dir + 'encoders/classification/dev_te.pkl', 'rb'))
    _df['dev_avg'] = te.transform(_df[['Developer']])

    # Impute missing values using KNN
    knn = pickle.load(open(project_dir + 'imputers/classification/dev_knn.pkl', 'rb'))

    _df['dev_avg'] = knn.transform(_df[['dev_avg']])
    _df['dev_avg'] = _df['dev_avg'].astype(float)

    return _df

def dev_preprocess_freq_enc(_df):
    # Convert to string
    _df['Developer'] = _df['Developer'].astype(str)
    _df['Developer'] = _df['Developer'].str.replace("'", "").str.strip('[]')

    ce = pickle.load(open(project_dir + 'encoders/classification/dev_ce.pkl', 'rb'))
    _df['dev_freq'] = ce.transform(_df[['Developer']])['Developer']

    return _df

def genres_preprocess_dummies(_df):
    # Convert the genres column to a list of strings
    _df['Genres'] = _df['Genres'].astype(str)
    _df['Genres'] = _df['Genres'].str.strip(
        '[]').str.replace("'", "").str.split(", ")

    # drop Games, Strategy, Entertainment from the Genres column
    _df['Genres'] = _df['Genres'].apply(
        lambda x: [genre for genre in x if genre not in ['Games', 'Strategy', 'Entertainment']])

    # Load saved genres dummy variables
    saved_dummies = pd.read_csv(project_dir + 'encoders/classification/genres.csv')

    # Get the genres that are not in the saved dummy variables
    other = [genre for genre in _df['Genres'].explode(
    ).unique() if genre not in saved_dummies.columns]

    # Replace the genres that are not in the saved dummy variables with 'infrequent'
    _df['Genres'] = _df['Genres'].apply(
        lambda x: ['infrequent' if genre in other else genre for genre in x])

    # Preprocess test data using the saved dummy variables
    genres = pd.get_dummies(_df['Genres'].apply(
        pd.Series).stack(), prefix="genre", dummy_na=False).sum(level=0)
    genres = genres.reindex(columns=saved_dummies.columns, fill_value=0)

    # Fill the dummy columns with 0 if nan
    genres = genres.fillna(0)

    # Add the dummy variables to the original dataframe
    _df = pd.concat([_df, genres], axis=1)

    # Fill the NaN values with 0
    genre_cols = [col for col in _df.columns if col.startswith(
        'genre')]  # get all columns with prefix 'genre'
    _df[genre_cols] = _df[genre_cols].fillna(0)  # fill the NaN values with 0

    return _df

def langs_preprocess_dummies(_df):
    # Convert the langs column to a list of strings
    _df['Languages'] = _df['Languages'].astype(str)
    _df['Languages'] = _df['Languages'].str.strip(
        '[]').str.replace("'", "").str.split(", ")

    # Create a column with the number of languages supported
    _df['langs_count'] = _df['Languages'].apply(lambda x: len(x))

    # drop English from the Languages column
    _df['Languages'] = _df['Languages'].apply(
        lambda x: [lang for lang in x if lang not in ['EN']])

    saved_dummies = pd.read_csv(project_dir + 'encoders/classification/langs.csv')

    # Get the languages that are not in the saved dummy variables
    other = [lang for lang in _df['Languages'].explode(
    ).unique() if lang not in saved_dummies.columns]

    # Replace the languages that are not in the saved dummy variables with 'infrequent'
    _df['Languages'] = _df['Languages'].apply(
        lambda x: ['infrequent' if lang in other else lang for lang in x])

    # Preprocess test data using the saved dummy variables
    langs = pd.get_dummies(_df['Languages'].apply(
        pd.Series).stack(), prefix="lang", dummy_na=False).sum(level=0)
    langs = langs.reindex(columns=saved_dummies.columns, fill_value=0)

    # Fill the dummy columns with 0 if nan
    langs = langs.fillna(0)

    # Add the dummy variables to the original dataframe
    _df = pd.concat([_df, langs], axis=1)

    # Fill NaN with 0
    lang_cols = [col for col in _df.columns if col.startswith(
        'lang')]  # get all columns with prefix 'lang'
    # fill NaN with 0 for selected columns
    _df[lang_cols] = _df[lang_cols].fillna(0)

    return _df

def purchases_preprocess(_df):
    # Convert the In-app Purchases column to a list of floats
    _df['In-app Purchases'] = _df['In-app Purchases'].astype(str)
    _df['In-app Purchases'] = _df['In-app Purchases'].str.strip(
        '[]').str.replace("'", "").str.split(", ")

    # Convert to float
    _df['In-app Purchases'] = _df['In-app Purchases'].apply(
        lambda x: [float(i) for i in x])

    # Get the number of in-app purchases
    _df['purchases_count'] = _df['In-app Purchases'].apply(lambda x: len(x))

    # Get the lowest, highest and average purchase
    _df['lowest_purchase'] = _df['In-app Purchases'].apply(
        lambda x: min(x) if len(x) > 0 else 0)
    _df['highest_purchase'] = _df['In-app Purchases'].apply(
        lambda x: max(x) if len(x) > 0 else 0)
    _df['average_purchase'] = _df['In-app Purchases'].apply(
        lambda x: np.mean(x) if len(x) > 0 else 0)

    _df['lowest_purchase'] = _df['lowest_purchase'].fillna(0)
    _df['highest_purchase'] = _df['highest_purchase'].fillna(0)
    _df['average_purchase'] = _df['average_purchase'].fillna(0)

    return _df

def age_preprocess(_df):
    # Convert to string
    _df['Age Rating'] = _df['Age Rating'].astype(str)

    # Remove the + sign
    _df['Age Rating'] = _df['Age Rating'].str.replace('+', '')

    # Convert to int
    _df['Age Rating'] = _df['Age Rating'].astype(float)

    # Impute missing values using simple imputer with median strategy
    simple_imputer = pickle.load(
        open(project_dir + 'imputers/classification/age_simple.pkl', 'rb'))

    _df['Age Rating'] = simple_imputer.transform(_df[['Age Rating']])
    return _df

def price_preprocess(_df):
    # Convert to float
    _df['Price'] = _df['Price'].astype(float)

    # fill the missing values with 0 (free)
    _df['Price'] = _df['Price'].fillna(0)

    return _df

def compute_excitement_score(text, _sia):
    # compute the polarity scores for the given text
    scores = _sia.polarity_scores(text)

    # compute the excitement score as the sum of the positive and negative polarity scores
    excitement_score = scores['pos'] + abs(scores['neg'])

    return excitement_score

def compute_attractive_score(text, tokenizer):
    # define a list of keywords that might make a game attractive to users
    attractive_keywords = ['graphics', 'gameplay', 'storyline', 'characters']

    # tokenize the text into words and count how many attractive keywords appear
    words = tokenizer(text.lower())

    num_attractive_keywords = len(
        [word for word in words if word in attractive_keywords])

    # compute the attractive score as the ratio of attractive keywords to total words
    attractive_score = num_attractive_keywords / \
        len(words) if len(words) > 0 else 0

    return attractive_score

def desc_preprocess(_df):
    _df['Description'] = _df['Description'].astype(str)

    # Create column for number of words in description
    _df['desc_word_count'] = _df['Description'].apply(lambda x: len(x.split()))

    sia_desc = pickle.load(open(project_dir + 'encoders/classification/sia_desc.pkl', 'rb'))
    tokenizer = pickle.load(
        open(project_dir + 'encoders/classification/desc_tokenizer.pkl', 'rb'))

    _df['excitement_score'] = _df['Description'].apply(
        lambda x: compute_excitement_score(x, sia_desc))
    _df['attractive_score'] = _df['Description'].apply(
        lambda x: compute_attractive_score(x, tokenizer))

    return _df

def name_preprocess(_df):
    _df['Name'] = _df['Name'].astype(str)

    # Create column for number of words in subtitle
    _df['name_word_count'] = _df['Name'].apply(
        lambda x: len(str(x).split(" ")))

    sia_name = pickle.load(open(project_dir + 'encoders/classification/sia_name.pkl', 'rb'))

    _df['name_sia'] = _df['Name'].apply(
        lambda x: compute_excitement_score(x, sia_name))

    return _df

def sub_preprocess(_df):
    _df['Subtitle'] = _df['Subtitle'].astype(str)

    # Create column for number of words in subtitle
    _df['sub_word_count'] = _df['Subtitle'].apply(
        lambda x: len(str(x).split(" ")))

    sia_sub = pickle.load(open(project_dir + 'encoders/classification/sia_sub.pkl', 'rb'))

    # _df['sub_sia'] = _df['Subtitle'].apply(lambda x: compute_excitement_score(x, sia_sub))

    return _df

def detect_objects(image_path):
    """
    Detect objects in an image and return the number of objects detected.

    https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18

    Parameters:
        image_path (str): The file path of the image to be analyzed.

    Returns:
        int: The number of objects detected in the image.
    """
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to highlight the edges of objects in the image
    edges = cv2.Canny(gray, 100, 200)

    # Apply a threshold to convert the edge map to a binary image
    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Return the number of objects detected
    return len(contours)

def preprocess_icon(img_path):
    # Load the game icon image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))

    # Extract color features using color histograms
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    color_features = []
    for i in range(3):
        hist = cv2.calcHist([img_lab], [i], None, [256], [0, 256])
        color_features.append(hist.ravel())
    color_features = np.concatenate(color_features)

    # Extract shape features using local binary patterns
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist_lbp, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    edge_features = hist_lbp.astype(float)

    # Combine the color and shape features into a single feature vector
    feature_vector = np.concatenate((color_features, edge_features))

    # Normalize the feature vector to have unit length
    normalized_feature_vector = feature_vector / np.linalg.norm(feature_vector)

    return normalized_feature_vector

def icons_preprocess(_df):
    # Create a list to store the feature vectors
    icon_features = []

    _df['Icon URL'] = _df['Icon URL'].astype(str)
    _df['icon_objects'] = np.nan

    # Iterate over the images and extract the features
    for i, row in tqdm(_df.iterrows(), total=_df.shape[0]):
        _df.loc[i, 'icon_objects'] = detect_objects(row['Icon URL'])
        feature_vec = preprocess_icon(row['Icon URL'])
        icon_features.append((row['Icon URL'], feature_vec))

    pca = pickle.load(open(project_dir + 'encoders/classification/icon_pca.pkl', 'rb'))

    reduced_features = pca.transform([f[1] for f in icon_features])

    # Convert the reduced features to a dataframe
    # Convert the reduced features to a dataframe
    icon_features_df = pd.DataFrame({'Icon URL': [f[0] for f in icon_features],
                                     'Icon1': reduced_features[:, 0],
                                     'Icon2': reduced_features[:, 1],
                                     'Icon3': reduced_features[:, 2],
                                     'Icon4': reduced_features[:, 3],
                                     'Icon5': reduced_features[:, 4],
                                     'Icon6': reduced_features[:, 5],
                                     'Icon7': reduced_features[:, 6],
                                     'Icon8': reduced_features[:, 7], })

    # Merge the icon features with the original dataframe on the icon URL
    _df = _df.merge(icon_features_df, on='Icon URL', how='left')

    return _df

def reviews_preprocess(data):
    # Apply sentiment_analysis

    sia_reviews = pickle.load(
        open(project_dir + 'encoders/classification/sia_reviews.pkl', 'rb'))

    # Only preprocess the reviews that are not null
    data['Reviews'] = data['Reviews'].apply(
        lambda x: [sia_reviews.polarity_scores(review)['compound'] for review in x] if isinstance(x, list) and len(
            x) > 0 else [])

    # Get the lowest, highest and average Reviews
    data['lowest_review'] = data['Reviews'].apply(
        lambda x: min(x) if len(x) > 0 else None)
    data['highest_review'] = data['Reviews'].apply(
        lambda x: max(x) if len(x) > 0 else None)

    # Calculate the average review without the outliers via z-score
    data['average_review'] = data['Reviews'].apply(
        lambda x: np.mean(x) if len(x) > 0 else None)

    # Impute missing values using KNN
    knn_low = pickle.load(
        open(project_dir + 'imputers/classification/review_low_knn.pkl', 'rb'))
    knn_high = pickle.load(
        open(project_dir + 'imputers/classification/review_high_knn.pkl', 'rb'))
    knn_avg = pickle.load(
        open(project_dir + 'imputers/classification/review_avg_knn.pkl', 'rb'))

    data['lowest_review'] = knn_low.transform(data[['lowest_review']])
    data['highest_review'] = knn_high.transform(data[['highest_review']])
    data['average_review'] = knn_avg.transform(data[['average_review']])

    return data

def size_preprocess(_df):
    # Convert to float
    _df['Size'] = _df['Size'].astype(float)
    
    # Apply log transformation using the natural logarithm function (log base e)
    _df['size_log'] = np.log(_df['Size'] / 1000000)
    
    # Impute missing values using KNN
    knn = pickle.load(open(project_dir + 'imputers/regression/size_knn.pkl', 'rb'))

    _df['size_log'] = knn.transform(_df[['size_log']])
    return _df

def user_count_preprocess(_df):
    # Convert to float
    _df['User Rating Count'] = _df['User Rating Count'].astype(float)
    
    # Impute missing values using KNN
    knn = pickle.load(open(project_dir + 'imputers/classification/user_count_knn.pkl', 'rb'))

    _df['User Rating Count'] = knn.transform(_df[['User Rating Count']])
    
    return _df

def test_pipeline(_df):
    # _df = download_icons(_df)
    _df = download_reviews(_df)
    
    _df = _df.drop(['Primary Genre', 'ID', 'URL'], axis=1)
    
    _df = date_preprocess(_df)
    _df = purchases_preprocess(_df)

    _df = age_preprocess(_df)
    _df = price_preprocess(_df)
    
    _df = name_preprocess(_df)
    _df = sub_preprocess(_df)
    _df = desc_preprocess(_df)
    # _df = icons_preprocess(_df, test)
    
    _df = _df.drop(['Name',
                    'Subtitle',
                    'Description',
                    'Icon URL',
                    'Languages',
                    'Genres',
                    'In-app Purchases',
                    'Original Release Date',
                    'Current Version Release Date'], axis=1)
    
    # Dropping all the columns that are already preprocessed as they are not needed anymore 
    # But more importantly for KNN imputation to work
    
    _df = size_preprocess(_df)
    _df = _df.drop(['Size'], axis=1)
    _df = user_count_preprocess(_df)
    
    _df = dev_preprocess_freq_enc(_df)
    _df = dev_preprocess_target_enc(_df)
    
    _df = _df.drop(['Developer'], axis=1)
    
    _df = reviews_preprocess(_df)
    _df = _df.drop(['Reviews'], axis=1)

    # Scaling
    scaler = pickle.load(open(project_dir + 'scalers/classification/std_scaler.pkl', 'rb'))

    # Get the feature names in the order they appear in the input DataFrame
    feature_names = _df.columns

    # Transform the data using the scaler object
    _df = scaler.transform(_df)
    _df = pd.DataFrame(_df, columns=feature_names)

    # Select the features
    selector = pickle.load(open(project_dir + 'encoders/classification/selector.pkl', 'rb'))
    _df = selector.transform(_df)
    
    return _df


# Define mapping dictionary
mapping = {
    0: 'Low',
    1: 'Intermediate',
    2: 'High'}


def run_model(model_name, data):
    if model_name == 'Random Forest':
        model = pickle.load(open(project_dir + 'models/classification/rf.pkl', 'rb'))
    elif model_name == 'XGBoost':
        model = pickle.load(open(project_dir + 'models/classification/xgb.pkl', 'rb'))
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
    # Preprocess the test data
    data = test_pipeline(data)
    
    # Remove the deployment reviews csv and the deployment icons contents if they exist
    if os.path.exists(project_dir + 'datasets/deploy/reviews.csv'):
        os.remove(project_dir + 'datasets/deploy/reviews.csv')
        
    if os.path.exists(project_dir + 'icons/deploy/0.png'):
        os.remove(project_dir + 'icons/deploy/0.png')

    # Predict the class and the probability
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    probability = probability[0][prediction[0]]

    # Map the results to their respective classes
    return mapping[prediction[0]], probability

