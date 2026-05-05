# Saudi Real Estate Rental Analytics Dashboard

An interactive data science dashboard for analyzing real estate rental prices across major Saudi cities using AQAR rental listing data.

## Project Overview

This project connects Data Science with Real Estate by analyzing rental property listings, identifying price trends, comparing cities and districts, and building machine learning models to estimate rental prices.

The dashboard was built using Python and Streamlit, with interactive charts and filters to help users explore the rental market visually.

## Objectives

- Analyze rental price trends across major Saudi cities.
- Compare districts based on average rental price.
- Calculate key real estate metrics such as average rental price, median rental price, average area, and price per square meter.
- Identify key factors affecting rental prices.
- Build and compare machine learning models for rental price prediction.
- Deploy the dashboard online using Streamlit Cloud and GitHub.

## Dataset

The project uses a real AQAR rental listing dataset from Kaggle.

The dataset includes property-related features such as:

- City
- District
- Property size
- Property age
- Bedrooms
- Bathrooms
- Living rooms
- Garage availability
- Furnishing status
- Duplex indicator
- Rental price

## Tools and Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- GitHub
- Streamlit Cloud

## Dashboard Features

- Interactive filters by city, district, and property type.
- KPI cards showing key market metrics.
- Market insight cards summarizing important findings.
- Top districts by average rental price.
- Area vs rental price analysis.
- Property type distribution.
- City-level rental market map.
- Machine learning model comparison.
- Feature importance analysis.
- Rental price prediction tool.
- Downloadable filtered dataset.

## Machine Learning

Two regression models were used and compared:

1. Linear Regression
2. Random Forest Regressor

Model performance was evaluated using:

- MAE: Mean Absolute Error
- R² Score

Random Forest was used for the final rental price prediction because it can better capture non-linear relationships between property features and rental price.

## Key Insights

The dashboard highlights:

- Which cities have higher average rental prices.
- Which districts are more expensive.
- How property size relates to rental price.
- The dominant property type in the dataset.
- The most important factors affecting rental prices based on the Random Forest model.

## Project Value

This project demonstrates how data science can support real estate decision-making by transforming raw property listings into useful insights. It combines data cleaning, exploratory data analysis, visualization, machine learning, and deployment into one complete portfolio project.

## Project Summary

Built and deployed an interactive Saudi real estate rental analytics dashboard using Python, Streamlit, Pandas, Plotly, and Scikit-learn. Analyzed real AQAR rental listing data to identify price trends across major Saudi cities, compare districts by average rental price and price per square meter, and predict rental prices using machine learning models.