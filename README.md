# games-rating-model
An ML powered model to predict games user ratings

## Project Overview
This project is a machine learning model that predicts the user rating of a game based on the game's features. The model is trained on a dataset of 5,214 games and their features. The model is trained using various regression and classification algorithms. The model is deployed using streamlit.

## Dataset
The dataset used for this project is a portion of a Kaggle dataset. The dataset is provided by the FCIS ML Team as a part of the FCIS ML course project. The dataset contains 5,214 games and their features.

The dataset used for this project can be found in the repository [here](Model/datasets/train/games-regression-dataset.csv) for regression and [here](Model/datasets/train/games-classification-dataset.csv) for classification. The original dataset can be found [here](https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games).

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

## Project Report
The project report can be found [here](Model/docs/report.pdf). The report contains how the project was developed, what were our approaches,
what features were used, how did we preprocess the data, the algorithms used, the results, and the conclusion.

## Results
### Regression Metrics

| Model | Train MSE | Val MSE | Test MSE | Train R2 | Val R2 | Test R2 | 
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost 			    | 0.1876 | 0.3264 | 0.4539 | 0.56 | 0.27 | 0.15 |
| GradientBoosting 	    | 0.1634 | 0.3209 | 0.4530 | 0.61 | 0.28 | 0.15 |
| PolynomialRegression 	| 0.2546 | 0.3384 | 0.4611 | 0.40 | 0.25 | 0.13 |
| ElasticNet 		    | 0.2805 | 0.3439 | 0.4641 | 0.34 | 0.23 | 0.13 |
| Linear Regression 	| 0.2795 | 0.3439 | 0.4644 | 0.34 | 0.23 | 0.13 |
| CatBoost 			    | 0.1644 | 0.3133 | 0.4675 | 0.61 | 0.30 | 0.12 |


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
