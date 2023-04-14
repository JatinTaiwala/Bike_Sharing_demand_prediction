# Helped A Bike Renting Service Predict Their Demand:
This is a linear regression model that predicts the demand for bike sharing for Boombikes, a bike-sharing company. The model uses historical bike rental data along with weather data to make accurate predictions.

## Installation
To install and run this project on your local machine, follow these steps:
git clone https://github.com/<JatinTaiwala>/Bike_sharing_demand_prediction.git
cd Bike_sharing_demand_prediction
pip install -r requirements.txt
jupyter notebook
## Usage
To use this project, follow the steps outlined in the Jupyter notebook. This will involve loading the necessary data, preprocessing the data, training the model, and making predictions. You can also modify the model parameters and retrain the model to improve its performance.

## Data Source
The data used in this project was provided by upgrad: Bike Sharing Demand.

## Model Evaluation
The performance of the model was evaluated using the Root Mean Squared Logarithmic Error (RMSLE) metric. The model achieved an RMSLE value of 0.346, indicating good performance.

Additionally, the R^2 values for the train and test datasets were calculated:

Train dataset R^2: 0.833
Test dataset R^2: 0.8038
Train dataset Adjusted R^2: 0.829
Test dataset Adjusted R^2: 0.7944
These values suggest that the linear regression model is a good predictor of bike sharing demand based on historical rental data and weather information, without overfitting the data.
## Results
The model was able to accurately predict bike rental demand for Boombikes based on historical rental data and weather information. The results show that the demand for bike rentals is higher on weekdays and during rush hour, and is also affected by weather conditions such as temperature, humidity, and windspeed.

## Contributors
This project was developed by Jatin Taiwala.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Future Work
Some potential ideas for future work on this project include:

Experimenting with different machine learning models and feature engineering techniques to improve performance.
Incorporating additional data sources such as traffic data to make more accurate predictions.
Developing a web application or API that allows users to interact with the model and make predictions.
