# Car Cost Prediction using Multiple Linear Regression

## Overview
This project aims to predict the prices of used cars using a Multiple Linear Regression (MLR) model. The project involves loading a dataset of used cars, performing data cleaning and feature engineering, exploring the data through exploratory data analysis (EDA), and finally training a regression model to estimate car prices based on various attributes.

## Directory Structure
```
CarCostPrediction-MLR/
│
├── data/
│   └── used_cars.csv              # The dataset containing used car listings
│
├── model/
│   └── CarCostPrediction_MLR.ipynb # Jupyter notebook containing the complete ML pipeline
│
└── README.md                      # Project documentation
```

## Dataset Description
The dataset `used_cars.csv` contains various details about used cars. The key features include:
- `brand`: The manufacturer of the car (e.g., Ford, BMW, Audi).
- `model`: The specific model of the car.
- `model_year`: The year the car was manufactured.
- `milage`: The total distance the car has traveled (in miles).
- `fuel_type`: The type of fuel the car consumes (e.g., Gasoline, Hybrid).
- `engine`: Engine specifications, which include horsepower, displacement (liters), and cylinders.
- `transmission`: The type of transmission (e.g., Automatic, Manual).
- `ext_col`: Exterior color of the car.
- `int_col`: Interior color of the car.
- `accident`: Whether any accidents have been reported.
- `clean_title`: Indicates if the car has a clean title.
- `price`: The target variable, representing the price of the used car.

## Key Steps Performed in the Notebook
1. **Data Loading and Inspection**: Loading the data using `pandas` and checking for missing values and general statistics.
2. **Data Cleaning**: Cleaning numeric columns like `price` and `milage` by removing string characters (like `$` and `mi.`) and casting them to float.
3. **Feature Engineering**: 
    - Extracting `horsepower`, `engine_liters`, and `cylinders` from the `engine` column.
    - Calculating `car_age` from `model_year`.
    - Categorizing transmission types and grouping rare brands into an 'Other' category.
4. **Exploratory Data Analysis (EDA)**: Visualizing the price distribution and the correlation between numerical features using `matplotlib` and `seaborn`.
5. **Modeling (MLR)**: Setting up a machine learning pipeline using `scikit-learn` with `ColumnTransformer` for applying `StandardScaler` to numerical features and `OneHotEncoder` to categorical features, followed by a `LinearRegression` model.
6. **Evaluation**: Evaluating the model's performance using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared ($R^2$).

## Requirements
To run the notebook, you need the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies via pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run
1. Clone the repository or download the project files.
2. Ensure you have the required dependencies installed.
3. Open a terminal or command prompt in the `model/` directory.
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `CarCostPrediction_MLR.ipynb` and run the cells sequentially to observe the data processing and model training steps.
