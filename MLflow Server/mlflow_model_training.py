import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 1. Load your dataset (Replace 'data.csv' with your actual file)
df = pd.read_csv('StudentsPerformance.csv')

# Define features and target
X = df.drop('math score', axis=1)
y = df['math score']

# 2. Preprocessing: Convert text categories into numbers
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)]
)

# 3. Create a Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MLflow Tracking
mlflow.set_tracking_uri("https://mlflow-server-614417916386.europe-west9.run.app/")
mlflow.set_experiment("Math_Score_Prediction2")

with mlflow.start_run():
    # Fit the model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions and calculate error
    predictions = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    
    # Log the actual model
    mlflow.sklearn.log_model(model_pipeline, "model")
    
    print(f"Run completed. MSE: {mse}")