# Personality Prediction Web Application

This project is a web application that predicts whether a person is an introvert or an extrovert based on their social behavior. The application is built using Streamlit and uses a machine learning model to make the predictions.

## Features

-   **Interactive Web Interface**: A user-friendly web interface built with Streamlit to input data and view predictions.
-   **Personality Prediction**: Predicts personality traits (Introvert or Extrovert) using a trained LightGBM model.
-   **Data Visualization**: Visualizes the prediction probabilities using Plotly charts.
-   **Downloadable Reports**: Allows users to download the prediction results.

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/personality-predictor.git
    cd personality-predictor
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```

This will open the application in your web browser.

## Model

The personality prediction model is a LightGBM classifier trained on a dataset of social behavior.

### Features

The model uses the following features to predict personality:

-   `Time_spent_Alone`
-   `Stage_fear`
-   `Social_event_attendance`
-   `Going_outside`
-   `Drained_after_socializing`
-   `Friends_circle_size`
-   `Post_frequency`

### Preprocessing

The data is preprocessed using the following steps:

-   **Imputation**: Missing values are imputed using the mean for numerical features and the most frequent value for categorical features.
-   **One-Hot Encoding**: Categorical features are converted to numerical features using one-hot encoding.
-   **SMOTE**: The training data is balanced using the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.

### Training

The model is trained using a GridSearchCV to find the best hyperparameters for the LightGBM classifier. The trained model is saved to `tuned_personality_model.pk2.0`.

## Data

The dataset used for training and testing is included in the repository:

-   `train.csv`: The training dataset.
-   `test.csv`: The testing dataset.
-   `sample_submission.csv`: A sample submission file in the correct format.
