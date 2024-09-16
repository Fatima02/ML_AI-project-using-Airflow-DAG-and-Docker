#AIRFLOW_IMAGE_NAME=apache/airflow:2.8.1
#AIRFLOW_UID=50000
FROM apache/airflow:2.8.1
COPY requirements.txt /requirements.txt
# Install the Python packages specified in requirements.txt
RUN pip install -r /requirements.txt