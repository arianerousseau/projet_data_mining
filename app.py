import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Option to specify header row
        has_header = st.checkbox("Does the CSV file have a header?", value=True)
        if has_header:
            header_row = st.number_input("Header row (0-indexed)", value=0, min_value=0, step=1)
        else:
            header_row = None
        # Skip rows input
        skip_rows = st.number_input("Number of rows to skip at the start of the file", value=0, min_value=0, step=1)
        # Separator input
        separator = st.text_input("Separator", value=",")
        # Read the file
        try:
            data = pd.read_csv(uploaded_file, header=header_row, sep=separator, skiprows=skip_rows)
            le = LabelEncoder()
            data["b"] = le.fit_transform(data["b"])
            st.session_state.df = data
            st.write("Data imported")
            st.write(data)
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Function to handle missing values
def handle_missing_values(df, method, axis=None):
    if method == 'Delete rows':
        return df.dropna(axis=0)
    elif method == 'Delete columns':
        return df.dropna(axis=1)
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        
        if numeric_cols.empty:
            st.warning("No numeric columns found for the selected method.")
            return df

        if method == 'Replace with mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'Replace with median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'Replace with mode':
            imputer = SimpleImputer(strategy='most_frequent')
        elif method == 'KNN Imputation':
            imputer = KNNImputer()
        else:
            return df
        
        if method == 'Replace with None':
            return df.fillna('None')
        
        df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        df_non_numeric = df[non_numeric_cols]
        
        return pd.concat([df_numeric, df_non_numeric], axis=1)

# Function to normalize data
def normalize_data(df, method):
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for normalization.")
        return df

    if method == 'Min-Max':
        scaler = MinMaxScaler()
    elif method == 'Z-score':
        scaler = StandardScaler()
    elif method == 'Robust Scaler':
        scaler = RobustScaler()
    elif method == 'Max Abs Scaler':
        scaler = MaxAbsScaler()
    else:
        return df
    
    df_numeric = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
    df_non_numeric = df[non_numeric_cols]
    
    return pd.concat([df_numeric, df_non_numeric], axis=1)

# Function to perform clustering
def perform_clustering(df, algorithm, params):
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        st.error("No numeric columns found for clustering.")
        return df
    if algorithm == 'K-Means':
        k = params['n_clusters']
        model = KMeans(n_clusters=k)
    elif algorithm == 'DBSCAN':
        eps = params['eps']
        min_samples = params['min_samples']
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        return df
    df['Cluster'] = model.fit_predict(df[numeric_cols])
    return df

# Function to perform prediction
def perform_prediction(df, algorithm, params, target_column):
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "Linear Regression":
        model = LinearRegression(**params)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(**params)
    elif algorithm == "Random Forest":
        model = RandomForestRegressor(**params)
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Return the results
    if algorithm in ["Linear Regression", "Random Forest"]:
        mse = mean_squared_error(y_test, predictions)
        return mse, predictions, y_test
    elif algorithm == "Logistic Regression":
        report = classification_report(y_test, predictions, output_dict=True)
        return report, predictions, y_test

# Function to visualize clusters
def visualize_clusters(df, algorithm):
    if 'Cluster' not in df.columns:
        st.warning("No clusters found. Please run a clustering algorithm first.")
        return
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to create a scatter plot.")
        return
    
    x_col = st.selectbox("Select X-axis feature", numeric_cols)
    y_col = st.selectbox("Select Y-axis feature", numeric_cols)
    if len(numeric_cols) > 2:
        z_col = st.selectbox("Select Z-axis feature (optional)", [None] + list(numeric_cols))
    else:
        z_col = None

    plt.figure(figsize=(10, 6))
    if z_col:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(df[x_col], df[y_col], df[z_col], c=df['Cluster'], cmap='viridis')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='Cluster', palette='viridis')
    
    st.pyplot(plt)

# Function to display cluster statistics
def cluster_statistics(df, algorithm):
    if 'Cluster' not in df.columns:
        st.warning("No clusters found. Please run a clustering algorithm first.")
        return
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("Number of data points in each cluster:")
    st.write(cluster_counts)

    if algorithm == 'K-Means':
        centers = df.groupby('Cluster').mean()
        st.write("Cluster centers:")
        st.write(centers)
    elif algorithm == 'DBSCAN':
        cluster_density = df.groupby('Cluster').apply(lambda x: len(x) / np.product(x[numeric_cols].max() - x[numeric_cols].min()))
        st.write("Cluster density:")
        st.write(cluster_density)

# Function to visualize predictions
def visualize_predictions(df, algorithm, predictions, target_column):
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), y, test_size=0.3, random_state=42)
    
    # Truncate or pad predictions to ensure the lengths match
    if len(predictions) > len(y_test):
        predictions = predictions[:len(y_test)]
    elif len(predictions) < len(y_test):
        predictions = np.pad(predictions, (0, len(y_test) - len(predictions)), mode='constant')
 
    # Ensure y_test and predictions are of the same type
    if algorithm == 'Logistic Regression':
        le = LabelEncoder()
        # Fit the encoder on the combined train and test set labels
        le.fit(list(y_train) + list(y_test) + list(predictions))
        y_test = le.transform(y_test)
        predictions = le.transform(predictions)
 
    if algorithm == 'Linear Regression':
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions)  # Ensure the predictions and y_test are of the same size
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        st.pyplot(plt)

    elif algorithm == 'Logistic Regression':
        cm = confusion_matrix(y_test, predictions)  # Ensure the predictions and y_test are of the same size
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='viridis')
        st.pyplot(plt)
 
# Function to display prediction statistics  
def prediction_statistics(algorithm, result):
    if algorithm == 'Linear Regression':
        st.write(f"Mean Squared Error: {result}")
    elif algorithm == 'Logistic Regression':
        st.write("Classification Report:")
        st.write(result)

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

st.title("Data Mining Project")
st.subheader("Capucine Foucher and Ariane Rousseau - BIA1")

# Create a sidebar for navigation
st.sidebar.title("MENU - Data Mining Project")
selection = st.sidebar.radio("Go to : ",  [
    "Part I: Initial Data Exploration",
    "Part II: Data Pre-processing and Cleaning",
    "Part III: Visualization of the cleaned data",
    "Part IV: Clustering or prediction",
    "Part V: Learning Evaluation"
])

# Initialize session state to store the dataframe
if "df" not in st.session_state:
    st.session_state.df = None

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

if selection == "Part I: Initial Data Exploration":
    
    st.header("Part I: Initial Data Exploration")
    
#---------------------------------------------------------------

    st.subheader("1.1. Data Loading")
    load_data()

#---------------------------------------------------------------

    st.subheader("1.2. Data Description")
    if st.session_state.df is not None:
        st.write("Preview of the first lines of the data:")
        st.dataframe(st.session_state.df.head())

        st.write("Preview of the last lines of the data:")
        st.dataframe(st.session_state.df.tail())
    else:
        st.warning("No data loaded. Please load a CSV file in the '1.1. Data Loading' section.")

#---------------------------------------------------------------

    st.subheader("1.3. Statistical Summary")
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.write("**Number of rows and columns:**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        
        st.write("**Column names:**")
        st.write(df.columns.tolist())
        
        missing_values = df.isnull().sum()
        st.subheader("Missing values per column")
        st.write(missing_values[missing_values > 0])
        
        st.write("**Basic statistical summary:**")
        st.dataframe(df.describe())
    else:
        st.warning("No data loaded. Please load a CSV file in the '1.1. Data Loading' section.")

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

elif selection == "Part II: Data Pre-processing and Cleaning":
    st.header("Part II: Data Pre-processing and Cleaning")
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Handle missing values
        st.write("**Handling Missing Values**")
        method = st.selectbox("Choose a method to handle missing values:",
                              ['Replace with None','Delete rows', 'Delete columns', 'Replace with mean', 'Replace with median', 'Replace with mode', 'Replace with KNN Imputation'])
        if method in ['Delete rows', 'Delete columns']:
            axis = 0 if method == 'Delete rows' else 1
            st.session_state.df = handle_missing_values(df, method, axis)
        else:
            st.session_state.df = handle_missing_values(df, method)
        st.write("Data after handling missing values:")
        st.dataframe(st.session_state.df)
#---------------------------------------------------------------
        # Normalize data
        st.write("**Normalizing Data**")
        normalize_method = st.selectbox("Choose a method to normalize the data:",
                                        ['None', 'Min-Max', 'Z-score', "Robust Scaler", "Max Abs Scaler"])
        if normalize_method != 'None':
            st.session_state.df = normalize_data(st.session_state.df, normalize_method)
        
        st.write("Data after normalization:")
        st.dataframe(st.session_state.df)
        
    else:
        st.warning("No data loaded. Please load a CSV file in the '1.1. Data Loading' section.")

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

elif selection == "Part III: Visualization of the cleaned data":
    st.header("Part III: Visualization of the cleaned data")
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        st.write("**Data Visualization**")
        plot_type = st.selectbox("Choose a type of plot:", ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap", "Bar Chart", "Pie Chart"])
        
        if plot_type == "Histogram":
            st.subheader("Histogram")
            column = st.selectbox("Select column for histogram", df.select_dtypes(include=['number']).columns)
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True)
            st.pyplot(plt)

        elif plot_type == "Box Plot":
            st.subheader("Box Plot")
            column = st.selectbox("Select column for box plot", df.select_dtypes(include=['number']).columns)
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[column])
            st.pyplot(plt)

        elif plot_type == "Scatter Plot":
            st.subheader("Scatter Plot")
            columns = st.multiselect("Select columns for scatter plot", df.select_dtypes(include=['number']).columns)
            if len(columns) == 2:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[columns[0]], y=df[columns[1]])
                st.pyplot(plt)
            else:
                st.warning("Please select exactly 2 columns for scatter plot.")
        elif plot_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

        elif plot_type == "Bar Chart":
            st.subheader("Bar Chart")
            object_columns = df.select_dtypes(include=['object']).columns
            if not object_columns.empty:
                column = st.selectbox("Select column for bar chart", object_columns)
                plt.figure(figsize=(10, 6))
                df[column].value_counts().plot(kind='bar')
                plt.xlabel(column)
                plt.ylabel("Count")
                st.pyplot(plt)
            else:
                st.warning("No non-numerical columns available for bar chart.")
        elif plot_type == "Pie Chart":
            st.subheader("Pie Chart")
            object_columns = df.select_dtypes(include=['object']).columns
            if not object_columns.empty:
                column = st.selectbox("Select column for pie chart", object_columns)
                plt.figure(figsize=(10, 6))
                df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel(column)
                st.pyplot(plt)
            else:
                st.warning("No non-numerical columns available for pie chart.")

    else:
        st.warning("No data loaded. Please load a CSV file in the '1.1. Data Loading' section.")

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

elif selection == "Part IV: Clustering or prediction":
    st.header("Part IV: Clustering or prediction")
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Standardization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.select_dtypes(include=['number']))

        # PCA for dimensionality reduction (optional)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        pca_data = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])

        task = st.selectbox("Choose a task:", ["Clustering", "Prediction"])
        
        if task == "Clustering":
            st.session_state.algorithm = st.selectbox("Choose a clustering algorithm:", ["K-Means", "DBSCAN"])
            
            if st.session_state.algorithm == "K-Means":
                k = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=2)
                kmeans = KMeans(n_clusters=k, random_state=0)
                cluster_labels = kmeans.fit_predict(scaled_data)
                pca_data['Cluster'] = cluster_labels
                st.session_state.clustered_df = pca_data

                # Visualize clusters
                st.subheader("Visualization of Clusters (PCA)")
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_data, palette='Set1', legend='full')
                st.pyplot(plt)

                st.subheader("Cluster Centers")
                st.write(pd.DataFrame(kmeans.cluster_centers_, columns=df.select_dtypes(include=['number']).columns))

            elif st.session_state.algorithm == "DBSCAN":
                eps = st.slider("Select the maximum distance between two samples (eps)", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
                min_samples = st.slider("Select the minimum number of samples in a neighborhood (min_samples)", min_value=2, max_value=10, value=5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(scaled_data)
                pca_data['Cluster'] = cluster_labels
                st.session_state.clustered_df = pca_data

                # Visualize clusters
                st.subheader("Visualization of Clusters (PCA)")
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_data, palette='Set1', legend='full')
                st.pyplot(plt)

                st.subheader("Number of clusters found by DBSCAN")
                st.write(len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0))
            
        elif task == "Prediction":
            st.session_state.prediction_algorithm = st.selectbox("Choose a prediction algorithm:", ["Linear Regression", "Logistic Regression", "Random Forest"])
            st.session_state.target_column = st.selectbox("Choose the target column:", df.columns)
            
            # Select feature columns
            feature_columns = st.multiselect("Choose feature columns:", [col for col in df.columns if col != st.session_state.target_column])
            
            if st.session_state.prediction_algorithm == "Linear Regression":
                fit_intercept = st.checkbox("Fit intercept", value=True)
                params = {"fit_intercept": fit_intercept}
                model = LinearRegression(**params)
            elif st.session_state.prediction_algorithm == "Logistic Regression":
                max_iter = st.number_input("Max iterations", value=100, min_value=10, step=10)
                solver = st.selectbox("Solver", ["lbfgs", "liblinear", "sag", "saga"])
                params = {"max_iter": max_iter, "solver": solver}
                model = LogisticRegression(**params)
            elif st.session_state.prediction_algorithm == "Random Forest":
                n_estimators = st.number_input("Number of estimators", value=100, min_value=10, step=10)
                max_depth = st.number_input("Max depth", value=None, min_value=1, step=1)
                params = {"n_estimators": n_estimators, "max_depth": max_depth}
                model = RandomForestRegressor(**params)

            
            if st.button("Run Prediction"):
                result, predictions, y_test = perform_prediction(df, st.session_state.prediction_algorithm, params, st.session_state.target_column)
                if result is not None:
                    st.session_state.prediction_result = result
                    st.session_state.predictions = predictions
                    if st.session_state.prediction_algorithm == "Linear Regression":
                        st.write(f"Mean Squared Error: {result}")
                    elif st.session_state.prediction_algorithm == "Logistic Regression":
                        st.write("Classification Report:")
                        st.write(result)
                    elif st.session_state.prediction_algorithm == "Random Forest":
                        st.write("Prediction complete.")
                        st.write(f"Mean Squared Error: {result}")

                    # Display predictions and actual values
                    predictions_df = pd.DataFrame({
                        "Actual": y_test,
                        "Predicted": predictions[:len(y_test)]  # Ensure the predictions and y_test are of the same size
                    })
                    st.write(predictions_df)
    else:
        st.warning("No data loaded. Please load a CSV file in the '1.1. Data Loading' section.")

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

elif selection == "Part V: Learning Evaluation":
    st.header("Part V: Learning Evaluation")

    st.subheader("5.1. Visualization of clusters")
    if 'clustered_df' in st.session_state and 'algorithm' in st.session_state and st.session_state.clustered_df is not None:
        clustered_df = st.session_state.clustered_df
        algorithm = st.session_state.algorithm
        
        visualize_clusters(clustered_df, algorithm)
        cluster_statistics(clustered_df, algorithm)
        
    else:
        st.warning("No clusters found. Please run a clustering algorithm in 'Part IV: Clustering or prediction' section.")
    
    st.subheader("5.2. Visualization of Predictions")
    if 'predictions' in st.session_state and 'prediction_algorithm' in st.session_state and 'prediction_result' in st.session_state and 'target_column' in st.session_state:
        predictions = st.session_state.predictions
        prediction_algorithm = st.session_state.prediction_algorithm
        prediction_result = st.session_state.prediction_result
        target_column = st.session_state.target_column
        df = st.session_state.df

        print(predictions)
        visualize_predictions(df, prediction_algorithm, predictions, target_column)
        prediction_statistics(prediction_algorithm, prediction_result)
        
    else:
        st.warning("No predictions found. Please run a prediction algorithm in 'Part IV: Clustering or prediction' section.")