from datetime import timedelta
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
from datetime import datetime

def load_data(**kwargs):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'csv_files', 'healthcare_dataset.csv')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    full_data = pd.read_csv(file_path)
    kwargs['ti'].xcom_push(key='full_data', value=full_data.to_dict())  # Push DataFrame as dict to XCom
    return full_data

def pre_processing_data(**kwargs):
    ti = kwargs['ti']
    full_data_dict = ti.xcom_pull(task_ids='Loading-data', key='full_data')
    full_data = pd.DataFrame(full_data_dict)

    sns.heatmap(full_data.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
    plt.title('Missing Data: Training Set')
    plt.show()

    full_data.dropna(inplace=True)
    duplicate_rows_df = full_data[full_data.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_df.shape)

    full_data = full_data.drop_duplicates()
    ti.xcom_push(key='full_data', value=full_data.to_dict())
    return full_data

def visualization_data(**kwargs):
    ti = kwargs['ti']
    full_data_dict = ti.xcom_pull(task_ids='PreProcessing-data', key='full_data')
    full_data = pd.DataFrame(full_data_dict)

    full_data['Age'].plot(kind='hist')
    plt.title('The Age of Data')

    full_data['Medical Condition'].value_counts()

    Male = full_data.loc[full_data['Gender'] == 'Male'].shape[0]
    Female = full_data.loc[full_data['Gender'] == 'Female'].shape[0]
    if pd.isna(Male):
        Male = 0
    if pd.isna(Female):
        Female = 0
    if Male == 0 and Female == 0:
        print("No data available for plotting.")
    else:
        plt.pie([Male, Female], labels=['Male', 'Female'], autopct='%.2f%%')
        plt.title('Gender Distribution')
        plt.show()
    return full_data

def making_data_categorical(**kwargs):
    from sklearn.preprocessing import LabelEncoder
    ti = kwargs['ti']
    full_data_dict = ti.xcom_pull(task_ids='PreProcessing-data', key='full_data')
    full_data = pd.DataFrame(full_data_dict)
    print("Columns in DataFrame:", full_data.columns)
    le = LabelEncoder()
    #Gender = pd.get_dummies(full_data['Gender'], drop_first = True) # drop_first prevents multi-collinearity
    #Blood_Type= pd.get_dummies(full_data['Blood Type'], drop_first = True)
    #full_data = pd.concat([full_data, Gender, Blood_Type], axis = 1)
    # Drop unecessary columns
    #full_data.drop(['Gender', 'Blood Type'], axis = 1, inplace = True)
    full_data['Gender'] = le.fit_transform(full_data['Gender'])
    full_data['Blood Type'] = le.fit_transform(full_data['Blood Type'])
    full_data['Hospital'] = le.fit_transform(full_data['Hospital'])
    full_data['Admission Type'] = le.fit_transform(full_data['Admission Type'])
    full_data['Medication'] = le.fit_transform(full_data['Medication'])
    full_data['Test Results'] = le.fit_transform(full_data['Test Results'])
    #full_data['Medical Condition'] = le.fit_transform(full_data['Medical Condition'])
    full_data.drop(['Name','Date of Admission','Discharge Date', 'Doctor',"Insurance Provider"], axis=1, inplace=True)

    ti.xcom_push(key='full_data', value=full_data.to_dict())
    return full_data

def split_data(**kwargs):
    ti = kwargs['ti']
    full_data_dict = ti.xcom_pull(task_ids='Transforming-data', key='full_data')
    full_data = pd.DataFrame(full_data_dict)
    
    # Debug print
    print(full_data.head())
    print(full_data.columns)
    
    if 'Medical Condition' not in full_data.columns:
        raise ValueError("Column 'Medical Condition' not found in the data frame")
    
    x = full_data.drop('Medical Condition', axis=1)
    y = full_data['Medical Condition']

    # Check if x and y are not empty
    if x.empty or y.empty:
        raise ValueError("One or both of the arrays (x or y) are empty")
    
    # Use the scikit-learn train_test_split function
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
    scaler = StandardScaler()
    x_trained_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Push results to XCom if needed
    ti.xcom_push(key='x_trained_scaled', value=x_trained_scaled.tolist())
    ti.xcom_push(key='x_test_scaled', value=x_test_scaled.tolist())
    ti.xcom_push(key='y_train', value=y_train.tolist())
    ti.xcom_push(key='y_test', value=y_test.tolist())
     #scale data to try and get better accuracy
    

def model_training(**kwargs):
    import joblib
    import os
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    ti = kwargs['ti']
    
    # Pulling data from XCom
    x_trained_scaled = np.array(ti.xcom_pull(task_ids='Splitting-data', key='x_trained_scaled'))
    y_train = np.array(ti.xcom_pull(task_ids='Splitting-data', key='y_train'))

    # Log the data for debugging
    print("x_trained_scaled:", x_trained_scaled)
    print("y_train:", y_train)

    # Train the KNeighborsClassifier model
    lreg = KNeighborsClassifier()
    lreg.fit(x_trained_scaled, y_train)

    # Define the directory and model filename
    directory = 'airflow/'
    model_filename = os.path.join(directory, 'lreg_model.pkl')

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the trained model to a file
    joblib.dump(lreg, model_filename)

    # Push the file path to XCom instead of the model object
    ti.xcom_push(key='lreg_model_path', value=model_filename)
    
    return model_filename  # Return the model file path


def model_testing(**kwargs):
    import joblib
    import numpy as np
    ti = kwargs['ti']
    
    # Pull the model file path from XCom
    model_filename = ti.xcom_pull(task_ids='Training-Model', key='lreg_model_path')
    
    # Log the model file path
    print(f"Model file path retrieved from XCom: {model_filename}")
    
    if model_filename is None:
        raise ValueError("Model file path not found in XCom. Check the Training-Model task.")
    
    # Load the trained model from the file
    try:
        lreg = joblib.load(model_filename)
        if lreg is None:
            raise ValueError("Loaded model is None. Check if the model was saved correctly.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from file {model_filename}: {str(e)}")
    
    # Pulling test data from XCom
    x_test_scaled = np.array(ti.xcom_pull(task_ids='Splitting-data', key='x_test_scaled'))
    
    # Log the test data
    print(f"x_test_scaled retrieved from XCom: {x_test_scaled}")
    
    if x_test_scaled is None:
        raise ValueError("Test data not found in XCom. Check the Splitting-data task.")
    
    # Make predictions
    try:
        y_pred_lreg = lreg.predict(x_test_scaled)
    except Exception as e:
        raise RuntimeError(f"Failed to make predictions with the loaded model: {str(e)}")
    
    # Convert predictions to list before pushing to XCom
    y_pred_lreg_list = y_pred_lreg.tolist()
    
    # Push predictions to XCom
    ti.xcom_push(key='y_pred_lreg', value=y_pred_lreg_list)
    
    return y_pred_lreg_list


def model_evaluation(**kwargs):
    ti = kwargs['ti']
    y_test = np.array(ti.xcom_pull(task_ids='Splitting-data', key='y_test'))
    y_pred_lreg = np.array(ti.xcom_pull(task_ids='Testing-Model', key='y_pred_lreg'))

    logreg_accuracy = round(accuracy_score(y_test, y_pred_lreg) * 100, 2)
    print('Classification Model')
    print('--'*30)
    print('Accuracy', logreg_accuracy, '%')

default_args = {
    'retries': 10,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'AI-project-using-DAG',
    default_args=default_args,
    description="""This project is about healthcare dataset used for developing and 
                testing healthcare predictive models. First the dataset has been preprocessed using
                EDA techniques, then it has been analysed as a multi-class classification 
                problem in which patients have been classified on the basis of their diseases. """,
    schedule_interval='@daily',
    start_date=datetime(2024,9,13),
)

Task_Load_data = PythonOperator(
    task_id='Loading-data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

Task_preprocessing_data = PythonOperator(
    task_id='PreProcessing-data',
    python_callable=pre_processing_data,
    provide_context=True,
    dag=dag,
)

Task_visualization_data = PythonOperator(
    task_id='Data-visualization',
    python_callable=visualization_data,
    provide_context=True,
    dag=dag,
)

Task_transform = PythonOperator(
    task_id='Transforming-data',
    python_callable=making_data_categorical,
    provide_context=True,
    dag=dag,
)

Task_split = PythonOperator(
    task_id='Splitting-data',
    python_callable=split_data,
    provide_context=True,
    dag=dag,
)

Task_training = PythonOperator(
    task_id='Training-Model',
    python_callable=model_training,
    provide_context=True,
    dag=dag,
)

Task_testing = PythonOperator(
    task_id='Testing-Model',
    python_callable=model_testing,
    provide_context=True,
    dag=dag,
)

Task_Evaluation = PythonOperator(
    task_id='Model-Evaluation',
    python_callable=model_evaluation,
    provide_context=True,
    dag=dag,
)

Task_Load_data >> Task_preprocessing_data >> Task_visualization_data >> Task_transform >> \
Task_split >> Task_training >> Task_testing >> Task_Evaluation
