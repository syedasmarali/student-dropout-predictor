# Student Dropout Prediction

This project focuses on contains a comprehensive analysis of factors influencing student dropout rates in secondary education. It includes demographic information, academic performance, and social conditions that may contribute to a student's likelihood of dropping out.
Logistic regression model has been implemented to predict the dropout percentage with the help of Python. Moreover, a streamlit dashboard has also been developed to visualize the prediction and overall predicted model performance.

## Project Overview

- **Independent Variables**:
  A total of 34 independent variables have been incorporated into the prediction analysis. 
  - School: Name of the school attended (e.g., MS).
  - Gender: Gender of the student (e.g., M for Male, F for Female).
  - Age: Age of the student.
  - Address: Type of residence (U for urban, R for rural).
  - Family_Size: Size of the family (GT3 for greater than 3, LE3 for less than or equal to 3).
  - Parental_Status: Living arrangement of parents (A for living together, T for living apart).
  - Mother_Education: Education level of the mother (0 to 4).
  - Father_Education: Education level of the father (0 to 4).
  - Mother_Job: Type of job held by the mother.
  - Father_Job: Type of job held by the father.
  - Reason_for_Choosing_School: Reason for selecting the school (e.g., course).
  - Guardian: Guardian of the student (e.g., mother).
  - Travel_Time: Time taken to travel to school (in minutes).
  - Study_Time: Weekly study hours (1 to 4).
  - Number_of_Failures: Number of past class failures.
  - School_Support: Whether the student receives extra educational support (yes/no).
  - Family_Support: Family provided educational support (yes/no).
  - Extra_Paid_Class: Participation in extra paid classes (yes/no).
  - Extra_Curricular_Activities: Involvement in extracurricular activities (yes/no).
  - Attended_Nursery: Attendance in nursery school (yes/no).
  - Wants_Higher_Education: Desire to pursue higher education (yes/no).
  - Internet_Access: Availability of internet at home (yes/no).
  - In_Relationship: Romantic relationship status (yes/no).
  - Family_Relationship: Quality of family relationships (scale 1 to 5).
  - Free_Time: Amount of free time after school (scale 1 to 5).
  - Going_Out: Frequency of going out with friends (scale 1 to 5).
  - Weekend_Alcohol_Consumption: Alcohol consumption on weekends (scale 1 to 5).
  - Weekday_Alcohol_Consumption: Alcohol consumption on weekdays (scale 1 to 5).
  - Health_Status: Health rating of the student (scale 1 to 5).
  - Number_of_Absences: Total number of absences from school.
  - Grade_1: Grade received in the first assessment.
  - Grade_2: Grade received in the second assessment.
  - Final_Grade: Final grade received (G3).
  - Dropped_Out: Indicator of whether the student has dropped out (True/False).

- **Data Preparation**: 
  - The dataset was cleaned and formatted to ensure proper use in logistic regression analysis.
  - Non-numeric variables were converted into numeric values to make them compatible with the model.

- **Logistic Regression Analysis**: 
  - A logistic regression model was developed to predict the student dropout percentage on the provided variables.

- **Streamlit Web App**: 
  - A dashboard was created using the Streamlit library.
  - The app allows for visualization of the regression model and provides an interface for predicting student dropout percentage.

## Features

- Data cleaning and formatting for compatibility with logistic regression analysis.
- Development of a regression model for accurate performance predictions.
- Interactive web app for data analysis and price prediction using Streamlit.

## Installation

- Clone the repository:
   ```bash
   git clone https://github.com/syedasmarali/student-dropout-predictor.git
   
- Navigate into the project directory:
  ```bash
  cd student-dropout-predictor

- Create a virtual environment (optional but recommended):
  ```bash
  python -m venv .venv

- Activate the virtual environment:
  - On Windows:
    ```bash
    python -m venv .venv
  - On macOS/Linux:
    ```bash
    Source .venv/bin/activate

- Install the required packages:
  ```bash
  pip install -r requirements.txt

- Run the streamlit app:
  ```bash
  streamlit run src/app.py


## Technologies Used

- Python: For data analysis and logistic regression model.
- Streamlit: For building the web app and creating the dashboard.
- Pandas, NumPy, Sci-Kit Learn: For data manipulation, training the mode and analysis.
