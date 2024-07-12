import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

############################################# PART 1 #############################################
def main():
    st.title("Data Exploration and Analysis App")

    # 1. Data Loading
    st.sidebar.header("Upload your CSV file")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # User inputs for delimiter and header
        delimiter = st.sidebar.text_input("Enter delimiter (e.g., ',' for CSV)", value=',')
        header_option = st.sidebar.selectbox("Does the file have a header?", ("Yes", "No"))
        header = 0 if header_option == "Yes" else None
        
        # Load data
        try:
            data = pd.read_csv(uploaded_file, delimiter=delimiter, header=header)
            st.success("File successfully loaded")
            
            # 2. Data Preview
            st.header("Part I: Initial Data Exploration")
            st.subheader("Data Preview")
            st.subheader("First 5 rows")
            st.write(data.head())
            st.subheader("Last 5 rows")
            st.write(data.tail())

            # 3. Statistical Summary
            st.header("Statistical Summary")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
            st.write("Column names:", data.columns.tolist())

            # Missing values
            missing_values = data.isnull().sum()
            st.subheader("Missing values per column")
            st.write(missing_values[missing_values > 0])
            
            # Basic statistics
            st.subheader("Basic Statistics")
            st.write(data.describe(include='all'))

############################################# PART 2.1 #############################################
            # 4. Data Pre-processing and Cleaning
            st.header("Part II: Data Pre-processing and Cleaning")
            st.subheader("2.1 - Handling missing values")

            missing_value_options = st.selectbox(
                "Choose a method to handle missing values",
                ["None", "Delete rows", "Delete columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN Imputation"]
            )

            if missing_value_options == "Delete rows":
                data = data.dropna()
                st.write("Rows with missing values have been deleted.")
            elif missing_value_options == "Delete columns":
                data = data.dropna(axis=1)
                st.write("Columns with missing values have been deleted.")
            elif missing_value_options in ["Replace with mean", "Replace with median", "Replace with mode"]:
                strategy = missing_value_options.split()[-1]
                numeric_cols = data.select_dtypes(include=['number']).columns
                imputer = SimpleImputer(strategy=strategy)
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                st.write(f"Missing values have been replaced with the {strategy} of the column.")
            elif missing_value_options == "KNN Imputation":
                numeric_cols = data.select_dtypes(include=['number']).columns
                imputer = KNNImputer()
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                st.write("Missing values have been imputed using KNN imputation.")

            if missing_value_options != "None":
                st.subheader("Data Preview After Cleaning")
                st.write(data.head())
                st.write(data.tail())


############################################# PART 2.2 #############################################
            # 5. Data Normalization
            st.subheader("2.2 - Data Normalization")

            normalization_options = st.selectbox(
                "Choose a method to normalize the data",
                ["None", "Min-Max Normalization", "Z-score Standardization", "Robust Scaler", "Max Abs Scaler"]
            )

            if normalization_options == "Min-Max Normalization":
                scaler = MinMaxScaler()
                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                st.write("Data has been normalized using Min-Max Normalization.")
            elif normalization_options == "Z-score Standardization":
                scaler = StandardScaler()
                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                st.write("Data has been standardized using Z-score Standardization.")
            elif normalization_options == "Robust Scaler":
                scaler = RobustScaler()
                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                st.write("Data has been scaled using Robust Scaler.")
            elif normalization_options == "Max Abs Scaler":
                scaler = MaxAbsScaler()
                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                st.write("Data has been scaled using Max Abs Scaler.")

            if normalization_options != "None":
                st.subheader("Data Preview After Normalization")
                st.write(data.head())
                st.write(data.tail())

############################################# PART 3 #############################################
        except Exception as e:
            st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    main()

