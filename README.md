# 🚗 Used Car Price Prediction

This project aims to predict the selling price of used cars using machine learning techniques. By analyzing various features of the cars, the model provides an estimated price, assisting buyers and sellers in making informed decisions.

---

# Problem Statement:

The problem statement for this project is to predict the price of a used car based on a set of features, such as the years used, mileage, engine,power,kilometers driven and no. of seats.

# Solution approach:

A machine learning model can be used to predict used car prices by considering a variety of factors. The model can be trained on a dataset of historical car sales data, and it can then be used to predict the price of a car based on its features.

# Observations :

## The following observations were made during the course of this project:

    * The manfacturer Location of the car is the most important features for predicting the price of a used car.
    * The no. of years the car used has a negative on the price.
    * The mileage of the car also has impact on the price.
    * The power of the car has impact on the price.
    * The engine of the car has impact on the price.
    * The no. of seats the car has impact on the price.
    * The Kilometers the car driven has small impact on the price

# Insights:

The insights from this project can be used by car dealerships, car buyers, and other businesses that are involved in the used car market. These insights can help these businesses to make more informed decisions about the pricing of used cars. For example, car dealerships can use these insights to set more competitive prices for their used cars. Car buyers can use these insights to get a better deal on a used car. And other businesses that are involved in the used car market can use these insights to improve their operations.# Price-Prediction-for-Used-Cars-Datascience-Project

This project uses machine learning to predict the price of a used car. The model is trained on a dataset of historical car sales data, and it can then be used to predict the price of a car based on its features.

## 📁 Dataset Overview

* **Source**: [train-data.csv](https://github.com/MAHFUZATUL-BUSHRA/Car_Price_Prediction/blob/main/train-data.csv)
* **Features**:

  * `name`: Car name
  * `year`: Year of manufacture
  * `km_driven`: Kilometers driven
  * `fuel`: Type of fuel used (Petrol, Diesel, etc.)
  * `seller_type`: Type of seller (Individual, Dealer, etc.)
  * `transmission`: Transmission type (Manual, Automatic)
  * `owner`: Ownership status (First owner, Second owner, etc.)
  * `mileage`: Mileage of the car
  * `engine`: Engine capacity
  * `max_power`: Maximum power of the car
  * `torque`: Torque of the car
  * `seats`: Number of seats
  * `selling_price`: Price at which the car is being sold (Target variable)

---

## 🧠 Project Workflow

1. **Data Preprocessing**:

   * Handling missing values
   * Converting categorical variables using one-hot encoding
   * Feature engineering (e.g., calculating car age)

2. **Model Training**:

   * Splitting the dataset into training and testing sets
   * Training models like Linear Regression and Random Forest Regressor

3. **Model Evaluation**:

   * Evaluating model performance using metrics such as R² score and RMSE

---

## 📈 Model Performance

| Model                   | R² Score | RMSE |
| ----------------------- | -------- | ---- |
| Linear Regression       | 0.76     | 1.25 |
| Random Forest Regressor | 0.91     | 0.85 |


---

## Open `Car_Price_Prediction.ipynb` and follow the steps.

---

## 🔧 Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* jupyter

---

## 📌 Future Enhancements

* Implement hyperparameter tuning using GridSearchCV
* Explore advanced models like XGBoost and CatBoost
* Deploy the model using Flask or Streamlit for user interaction

---




