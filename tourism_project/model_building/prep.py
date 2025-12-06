# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Abhilashu/tourism-project/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from HF.")

# create a copy of dataset for preprocessing
df = tourism_dataset.copy()

# -------------------- DATA CLEANING AND PREPROCESSING BEGIN ------------------ #

#Step 1:
#Drop unnamed column and CustomerID column
df = df.drop(columns=["Unnamed: 0", "CustomerID"])


# Step 2:
# Strip string columns
#standardize text entries by removing any extra spaces from the beginning or end of each string
for col in df.select_dtypes(include="object").columns:
  df[col] = df[col].astype(str).str.strip()


# Step 3:
#Let us handle the numeric data type for having any unwanted values such as 'N/A' or 'unknown'
# so that our script doesnot break and we gracefully handle such values with nan.

num_cols = [
        "Age","CityTier","DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
        "PreferredPropertyStar","NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting",
        "MonthlyIncome","ProdTaken"
    ]
for c in num_cols:
  if c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# Step 4:
#Similarly, Let us handle our target variable to be only int 0/1
df["ProdTaken"] = df["ProdTaken"].fillna(0).astype(int)


# Step 5:
# From our excercise above let us handle the Gender and MartialStatus column

# Column Gender needs to be corrected for value 'Fe Male' to 'Female'.
# Similarly, Column MaritalStatus needs to be corrected for value 'Unmarried' changed to 'Single'

df["Gender"] = df["Gender"].replace("Fe Male", "Female")
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")


# Step 6:
#Handle for the negative value for column which clearly be the case of wrong data entry, e.g Age cannot be negative

neg_column_gaurd = [
        "Age","DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
        "PreferredPropertyStar","NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting",
        "MonthlyIncome"
    ]

for c in neg_column_gaurd:
  if c in df.columns:
    df.loc[df[c] < 0, c] = np.nan


# Step 7:
#Let us now handle missing values (represented as NaN) in dataset
# by imputing them based on whether a column is numerical or categorical.

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

for c in num_cols:
  df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
  if df[c].isna().any():
    df[c] = df[c].fillna(df[c].mode().iloc[0])

#---------------- DATA CLEANING & PREPROCESSING END ------------------ #


# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
        "Age","CityTier","DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
        "PreferredPropertyStar","NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting",
        "MonthlyIncome"
    ]

# List of categorical features in the dataset
categorical_features = [
        "TypeofContact","Occupation","Gender","MaritalStatus","Designation","ProductPitched"
    ]

# Define predictor dataframe (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,    # Ensures reproducibility by setting a fixed random seed
    stratify=y         # Preserves the class distribution in the split
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

try:
    for file_path in files:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],  # just the filename
            repo_id="Abhilashu/tourism-project",
            repo_type="dataset",
        )
    print("All files uploaded successfully to Hugging Face Hub.")
except HfHubHTTPError as e:
    print(f"Error uploading files to Hugging Face Hub: {e}")
except Exception as e:
    print(f"An unexpected error occurred during file upload: {e}")

