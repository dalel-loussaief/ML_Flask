import pandas as pd
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Charger le fichier Excel en spécifiant le séparateur décimal
data = pd.read_csv('Property Prices in Tunisia.csv')

print(data.head())

data.shape

data.isna().sum()

data.info()

data.head()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,10))
sns.countplot(y='city', data=data, order=data.city.value_counts().index)

plt.figure(figsize=(15,10))
sns.countplot(y='category', data=data, order=data.category.value_counts().index)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with numerical features
# Replace df with your actual DataFrame

# Create a box plot for each numerical feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[["room_count","bathroom_count"]])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.title('Box Plot of room_count Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with numerical features
# Replace df with your actual DataFrame

# Create a box plot for each numerical feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[["size"]])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.title('Box Plot of size Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with numerical features
# Replace df with your actual DataFrame

# Create a box plot for each numerical feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=data["price"])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.title('Box Plot of room_count Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()

import pandas as pd

# Assuming df is your DataFrame with numerical features
# Replace df with your actual DataFrame

# Define a function to remove outliers based on the IQR method
def remove_outliers(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

import numpy as np

# List of numerical features for which you want to remove outliers
numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()



# Remove outliers from numerical features
df_no_outliers = remove_outliers(data, numerical_features)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with numerical features
# Replace df with your actual DataFrame

# Create a box plot for each numerical feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_no_outliers[["room_count","bathroom_count"]])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.title('Box Plot of room_count and bathroom_count Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()

def preprocess_inputs(df, categories_to_exclude):
    df = df.copy()

    # Encode missing values properly
    df = df.replace(-1, np.NaN)
    # Fill missing values with column medians
    for column in ['room_count', 'bathroom_count', 'size']:
        df[column] = df[column].fillna(df[column].median())
    # Binary encoding
    df['type'] = df['type'].replace({'À Louer': 0, 'À Vendre': 1})
    # One-hot encoding
    for column in ['category', 'city']:
        if column not in categories_to_exclude:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
    df = df.drop('region', axis=1)
    df = df.drop('log_price', axis=1)

    return df



categories_to_exclude = ['Terrains et Fermes', 'Magasins, Commerces et Locaux industriels', 'Colocations']
print("Categories to exclude:", categories_to_exclude)

cleaned_df = preprocess_inputs(df_no_outliers, categories_to_exclude)
print("Columns after preprocessing:", cleaned_df.columns)

print(cleaned_df.columns)

cleaned_df.head()

cleaned_df.shape

import seaborn as sns
import matplotlib.pyplot as plt


# Calculate the correlation matrix
correlation_matrix = cleaned_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(36, 36))
price_correlations = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlations.head(10))

y = cleaned_df['price']
X = cleaned_df.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Utiliser Random Forest pour la régression
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# Make predictions on the training and testing data
y_pred_train = model_rf.predict(X_train)
y_pred_test = model_rf.predict(X_test)

# Calculate Mean Squared Error and R^2 Score for training data
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Calculate Mean Squared Error and R^2 Score for testing data
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print('Random Forest - Training Mean Squared Error:', mse_train)
print('Random Forest - Training R^2 Score:', r2_train)
print('Random Forest - Testing Mean Squared Error:', mse_test)
print('Random Forest - Testing R^2 Score:', r2_test)



