# Data Collection, Cleaning & Preprocessing

## Project Overview

This project focuses on preparing logistics shipment data for further analysis and machine learning.
The project covers data inspection, missing-value treatment, outlier handling, feature transformation, and data normalization using Python.

## Project Objectives

- Inspect and understand the logistics shipment dataset.
- Identify and handle missing values.
- Detect and treat outliers using the IQR method.
- Transform categorical data where required.
- Normalize numerical features using Min-Max scaling.
- Prepare a clean and consistent dataset for further analysis and modeling.

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn

## Data Preprocessing Steps

### 1. Data Inspection

The dataset was inspected to understand:

- Number of records and columns
- Data types
- Missing values
- Duplicate records
- Basic statistical characteristics

### 2. Missing Value Treatment

Missing values were identified and handled using appropriate preprocessing techniques.

For numerical fields, missing values were treated using statistical imputation where required.

### 3. Outlier Detection & Treatment

The Interquartile Range (IQR) method was used to identify potential outliers.

The IQR is calculated as:

```text
IQR = Q3 - Q1
