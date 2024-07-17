import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


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
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical columns for one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

    X = preprocessor.fit_transform(X)

    # Encode target column if it is categorical for classification tasks
    if algorithm == 'Logistic Regression':
        if y.nunique() > 10:
            st.error("Logistic Regression requires a categorical target variable with fewer unique values.")
            return None, None
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        st.session_state.label_encoder = label_encoder  # Save the label encoder for inverse transform
    elif algorithm == 'Linear Regression':
        if y.dtype == 'object':
            st.error("Linear Regression requires a numeric target variable.")
            return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Logistic Regression':
        model = LogisticRegression(max_iter=params['max_iter'])
    else:
        return None, None

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if algorithm == 'Linear Regression':
        error = mean_squared_error(y_test, predictions)
        return error, predictions
    elif algorithm == 'Logistic Regression':
        report = classification_report(y_test, predictions, output_dict=True)
        return report, predictions

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
    _, y_test, _, _ = train_test_split(df.drop(columns=[target_column]), y, test_size=0.3, random_state=42)
    
    if algorithm == 'Linear Regression':
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        st.pyplot(plt)
    elif algorithm == 'Logistic Regression':
        if 'label_encoder' in st.session_state:
            y_test = st.session_state.label_encoder.transform(y_test)
        cm = confusion_matrix(y_test, predictions)
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
st.subheader("Capucine Foucher - BIA1")

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
        
        task = st.selectbox("Choose a task:", ["Clustering", "Prediction"])
        
        if task == "Clustering":
            st.session_state.algorithm = st.selectbox("Choose a clustering algorithm:", ["K-Means", "DBSCAN"])
            
            if st.session_state.algorithm == "K-Means":
                n_clusters = st.number_input("Number of clusters (k):", min_value=2, max_value=10, value=3)
                params = {'n_clusters': n_clusters}
            elif st.session_state.algorithm == "DBSCAN":
                eps = st.number_input("Epsilon (eps):", min_value=0.1, max_value=10.0, value=0.5)
                min_samples = st.number_input("Minimum samples:", min_value=1, max_value=10, value=5)
                params = {'eps': eps, 'min_samples': min_samples}
            
            if st.button("Run Clustering"):
                df = perform_clustering(df, st.session_state.algorithm, params)
                st.write("Clustering results:")
                st.dataframe(df)
        
        elif task == "Prediction":
            st.session_state.prediction_algorithm = st.selectbox("Choose a prediction algorithm:", ["Linear Regression", "Logistic Regression"])
            st.session_state.target_column = st.selectbox("Choose the target column:", df.columns)
            
            if st.session_state.prediction_algorithm == "Logistic Regression":
                max_iter = st.number_input("Maximum iterations:", min_value=100, max_value=1000, value=200)
                params = {'max_iter': max_iter}
            else:
                params = {}
            
            if st.button("Run Prediction"):
                result, predictions = perform_prediction(df, st.session_state.prediction_algorithm, params, st.session_state.target_column)
                if result is not None:
                    st.session_state.prediction_result = result
                    st.session_state.predictions = predictions
                    if st.session_state.prediction_algorithm == "Linear Regression":
                        st.write(f"Mean Squared Error: {result}")
                    elif st.session_state.prediction_algorithm == "Logistic Regression":
                        st.write("Classification Report:")
                        st.write(result)
    else:
        st.warning("No data loaded. Please load a CSV file in the '1.1. Data Loading' section.")


#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

elif selection == "Part V: Learning Evaluation":
    st.header("Part V: Learning Evaluation")

    st.subheader("5.1. Visualization of clusters")
    if 'df' in st.session_state and st.session_state.df is not None:
        if 'algorithm' in st.session_state:
            visualize_clusters(st.session_state.df, st.session_state.algorithm)
        else:
            st.warning("Please run a clustering algorithm first.")

    st.subheader("5.2. Cluster statistics")
    if 'df' in st.session_state and st.session_state.df is not None:
        if 'algorithm' in st.session_state:
            cluster_statistics(st.session_state.df, st.session_state.algorithm)
        else:
            st.warning("Please run a clustering algorithm first.")
    
    st.subheader("5.3. Visualization of predictions")
    if 'predictions' in st.session_state and 'target_column' in st.session_state and 'prediction_algorithm' in st.session_state:
        visualize_predictions(st.session_state.df, st.session_state.prediction_algorithm, st.session_state.predictions, st.session_state.target_column)
    else:
        st.warning("Please run a prediction algorithm first.")

    st.subheader("5.4. Prediction statistics")
    if 'prediction_result' in st.session_state and 'prediction_algorithm' in st.session_state:
        prediction_statistics(st.session_state.prediction_algorithm, st.session_state.prediction_result)
    else:
        st.warning("Please run a prediction algorithm first.")