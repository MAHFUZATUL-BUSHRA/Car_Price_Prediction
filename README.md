# 🚗 Used Car Price Prediction

This project aims to predict the selling price of used cars using machine learning techniques. By analyzing various features of the cars, the model provides an estimated price, assisting buyers and sellers in making informed decisions.

---

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




