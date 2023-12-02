import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def read_custom_csv(file_path):
    """
    Read and correct the format of a CSV file.

    @param file_path: Path to the CSV file.
    @return: A pandas DataFrame with corrected data format.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip().strip('"').split(',')
            data.append(cleaned_line)
    df = pd.DataFrame(data[1:], columns=data[0])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given data series.

    @param series: Pandas Series containing stock prices.
    @param period: The number of periods to use for RSI calculation.
    @return: Pandas Series containing the RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_custom_features(df):
    """
    Add custom features such as moving averages and RSI to the DataFrame.

    @param df: Pandas DataFrame containing stock market data.
    @return: Pandas DataFrame with additional features.
    """
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

def prepare_data(df, target_col='Close'):
    """
    Prepare data for modeling by splitting into features and target.

    @param df: Pandas DataFrame containing the dataset.
    @param target_col: Name of the target column.
    @return: Tuple containing split training and testing datasets.
    """
    features = df[['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'RSI']]
    target = df[target_col]
    return train_test_split(features, target, test_size=0.2, random_state=0)

def train_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Random Forest Regressor model.

    @param X_train: Training data features.
    @param X_test: Testing data features.
    @param y_train: Training data target.
    @param y_test: Testing data target.
    @return: Tuple of the trained model and its RMSE on the test set.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, np.sqrt(mse)

# Updated plot_data function for raw data
def plot_data(df, title):
    """
    Plot the raw data of stock prices.

    @param df: DataFrame containing the stock data.
    @param title: Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title(f'{title} - Raw Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Updated plot_training_validation_data function for training/validation data
def plot_training_validation_data(df, X_train, y_train, X_test, y_test, title):
    """
    Plot the training and validation data for stock prices.

    @param df: Full DataFrame containing the stock data.
    @param X_train: Training feature data (pandas DataFrame).
    @param y_train: Training target data (pandas Series).
    @param X_test: Validation feature data (pandas DataFrame).
    @param y_test: Validation target data (pandas Series).
    @param title: Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(df.loc[X_train.index, 'Date'], y_train, color='blue', label='Training Data')
    plt.scatter(df.loc[X_test.index, 'Date'], y_test, color='orange', label='Validation Data')
    plt.title(f'{title} - Training & Validation Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def plot_raw_data(df, title='Stock Price Data'):
    """
    Plot the raw data of stock prices.

    Args:
    df (pandas.DataFrame): DataFrame containing the stock data with 'Date' and 'Close' columns.
    title (str): Title of the plot.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# File paths for your CSV files
file_paths = ['db/Stocks/bwen.csv', 'db/Stocks/cclp.csv', 'db/Stocks/iac.csv']

# Reading and processing each file
dataframes = {file_path.split('/')[-1].split('.')[0]: read_custom_csv(file_path) for file_path in file_paths}

# Adding custom features, training models, and plotting results
models_results = {}
for key, df in dataframes.items():
    # Plotting raw data for each stock
    plot_raw_data(df, f"{key.upper()} Stock - Raw Data")
    
    enhanced_df = add_custom_features(df.copy())
    X_train, X_test, y_train, y_test = prepare_data(enhanced_df, target_col='Close')
    model, rmse = train_evaluate_model(X_train, X_test, y_train, y_test)
    predictions = model.predict(X_test)

    # Plotting raw data and training/validation data
    plot_data(df, f"{key.upper()} Stock")
    plot_training_validation_data(enhanced_df, X_train, y_train, X_test, y_test, f"{key.upper()} Stock")

    models_results[key] = {'model': str(model), 'rmse': rmse}

# Outputting formatted results
print("Model Results:")
for stock, result in models_results.items():
    print(f"\nStock: {stock.upper()}")
    print(f"Model Type: {result['model']}")
    print(f"Root Mean Squared Error (RMSE): {result['rmse']:.4f}")