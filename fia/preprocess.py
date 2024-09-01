import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_with_labelencoder(df: pd.DataFrame, col_label: str):
    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=["object", "category"]).columns
    numerical_features = df.select_dtypes(include=["number"]).columns

    categorical_features = [
        features for features in categorical_features if features != col_label
    ]
    numerical_features = [
        features for features in numerical_features if features != col_label
    ]

    # Initialize dictionaries to store the encoders and scaler
    label_encoders = {}
    scaler = StandardScaler()

    # Encode categorical features using LabelEncoder
    for col in categorical_features:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Scale numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, label_encoders, scaler
