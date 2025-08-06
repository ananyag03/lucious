# Flight Delay & Cancellation Risk Analysis

This project analyzes flight data to predict high-risk flights (delays over 3 hours or cancellations) using logistic regression.

## Features

- Data cleaning and preprocessing
- Feature engineering (time of day, day of week, etc.)
- Logistic regression model for risk prediction
- Visualizations: confusion matrix, ROC curve, feature importance

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your `data.csv` file in the project directory.
   - Ensure it contains columns: `Airline`, `Departure`, `Arrival`, `delay_time`, `flight_date`, `std_hour`.

3. **Run the analysis:**
   ```sh
   python extract_data.py
   ```

## Outputs

- `confusion_matrix_logistic_regression.png`
- `roc_curve.png`
- `feature_importance_logistic.png`

## Notes

- The script expects `delay_time` to be numeric or `'Cancelled'`.
- `std_hour` should be an integer hour (0-23) for time-of-day