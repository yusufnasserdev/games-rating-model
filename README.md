# games-rating-model
An ML powered model to predict games user ratings

## Project Overview
This project is a machine learning model that predicts the user rating of a game based on the game's features. The model is trained on a dataset of 5,214 games and their features. The model is trained using various regression and classification algorithms. The model is deployed using streamlit.

## Dataset
The dataset used for this project is a portion of a Kaggle dataset. The dataset is provided by the FCIS ML Team as a part of the FCIS ML course project. The dataset contains 5,214 games and their features.

The original dataset can be found [here](https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games).

The dataset used for this project can be found in the repository [here](Model/datasets/train/games-regression-dataset.csv) for regression and [here](Model/datasets/train/games-classification-dataset.csv) for classification.

## Features

- Name: The name of the game
- Subtitle: The subtitle of the game
- Price: The price of the game
- Primary Genre: The primary genre of the game
- Genres: The genres of the game
- Languages: The languages of the game
- Size: The size of the game
- Original Release Date: The original release date of the game
- Current Version Release Date: The current version release date of the game
- Age Rating: The age rating of the game
- URL: The URL of the game
- Icon URL: The icon URL of the game
- Description: The description of the game
- Developer: The developer of the game
- User Rating Count: The user rating count of the game
- In-app Purchases: The in-app purchases of the game
- Rate: The rating of the game (Target for classification)
- Average User Rating: The average user rating of the game (Target for regression)


## Results
### Regression Metrics

| Model | Train MSE | Val MSE | Test MSE | Train R2 | Val R2 | Test R2 | 
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.1964 | 0.3246 | 0.4603 | 0.54 | 0.28 | 0.14 |
| GradientBoosting | 0.1766 | 0.3242 | 0.4610 | 0.58 | 0.28 | 0.14 |
| PolynomialRegression | 0.2630 | 0.3245 | 0.4643 | 0.38 | 0.28 | 0.13 |
| ElasticNet | 0.2805 | 0.3429 | 0.4709 | 0.34 | 0.24 | 0.12 |
| Linear Regression | 0.2805 | 0.3428 | 0.4711 | 0.34 | 0.24 | 0.12 |
| CatBoost | 0.1873 | 0.3201 | 0.4685 | 0.56 | 0.29 | 0.12 |


### Classification Metrics

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| SVC | 73.21% | 68.12% | 66.31% |
| RandomForest | 74.48% | 66.29% | 65.85% |
| LogisticRegression | 71.11% | 66.48% | 65.23% |
| CatBoost | 74.71% | 68.48% | 63.38% |

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Acknowledgements
- [FCIS ML Team](mailto:fcismlteam@cis.asu.edu.eg): Provided the dataset (Originally a portion from a Kaggle dataset) and the guidance while developing the project. 

## Team
- [Yusuf Nasser](https://github.com/yusufnasserdev)
- [Ayman Mohammed](https://github.com/I-Man104)
- [Samy Samer](https://github.com/samySamer)
- [Fatma Awad](https://github.com/fatmaawad)
- [Youssef Gad](https://github.com/YoussefAbdellatif)
