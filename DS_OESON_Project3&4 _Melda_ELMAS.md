# OESON PROJECT 3&4  Graduation Prediction


## Project Description

This project involves the exploration and analysis and prediction of student data obtained from a reputed university that include relevant information of the students currently enrolled in  the university for a specific program like personal information and curriculum data.

As a Data Scientist I performed EDA on the data and suggested a prediction model which predicts whether a college student will graduate or not using at least six machine learning algorithms.



```python
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
from matplotlib import style
from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
```

We need to dowload all necesary libraries for analysis and modeling of the data.

For reading our data set as csv file we used pandas library.


```python
df = pd.read_csv("Student_academic.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>10</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>28</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4419</th>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>5</td>
      <td>12.666667</td>
      <td>0</td>
      <td>15.5</td>
      <td>2.8</td>
      <td>-4.06</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4420</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>11.000000</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>4421</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>9</td>
      <td>1</td>
      <td>13.500000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>4422</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>12.000000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4423</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>12.7</td>
      <td>3.7</td>
      <td>-1.70</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>4424 rows × 35 columns</p>
</div>



We can see the whole dataset with all details of the students the data file has 4424 rows and 35 columns .
With the help of head and tail commends we can have an idea about the data which we are concentrating on "Target".


```python
df = pd.read_csv("Student_academic.csv")
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>10</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>28</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4419</th>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>5</td>
      <td>12.666667</td>
      <td>0</td>
      <td>15.5</td>
      <td>2.8</td>
      <td>-4.06</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4420</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>11.000000</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>4421</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>9</td>
      <td>1</td>
      <td>13.500000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>4422</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>12.000000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4423</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>12.7</td>
      <td>3.7</td>
      <td>-1.70</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



Okay, now we want to learn more about our data types are there any objects or all of them are numerical values?

Exploratory Data Analysis (EDA) it is crutial to have the idea what kind of data we are working for, then we are able to do feature engineering accordinly, after all we can then go to the modelling part.

## Exploratory Data Analysis (EDA) 


```python
#Checking the column information of the data frame

df.dtypes
```




    Marital status                                      int64
    Application mode                                    int64
    Application order                                   int64
    Course                                              int64
    Daytime/evening attendance                          int64
    Previous qualification                              int64
    Nacionality                                         int64
    Mother's qualification                              int64
    Father's qualification                              int64
    Mother's occupation                                 int64
    Father's occupation                                 int64
    Displaced                                           int64
    Educational special needs                           int64
    Debtor                                              int64
    Tuition fees up to date                             int64
    Gender                                              int64
    Scholarship holder                                  int64
    Age at enrollment                                   int64
    International                                       int64
    Curricular units 1st sem (credited)                 int64
    Curricular units 1st sem (enrolled)                 int64
    Curricular units 1st sem (evaluations)              int64
    Curricular units 1st sem (approved)                 int64
    Curricular units 1st sem (grade)                  float64
    Curricular units 1st sem (without evaluations)      int64
    Curricular units 2nd sem (credited)                 int64
    Curricular units 2nd sem (enrolled)                 int64
    Curricular units 2nd sem (evaluations)              int64
    Curricular units 2nd sem (approved)                 int64
    Curricular units 2nd sem (grade)                  float64
    Curricular units 2nd sem (without evaluations)      int64
    Unemployment rate                                 float64
    Inflation rate                                    float64
    GDP                                               float64
    Target                                             object
    dtype: object




```python
#Checking the null count of the data an the number of data.
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4424 entries, 0 to 4423
    Data columns (total 35 columns):
     #   Column                                          Non-Null Count  Dtype  
    ---  ------                                          --------------  -----  
     0   Marital status                                  4424 non-null   int64  
     1   Application mode                                4424 non-null   int64  
     2   Application order                               4424 non-null   int64  
     3   Course                                          4424 non-null   int64  
     4   Daytime/evening attendance                      4424 non-null   int64  
     5   Previous qualification                          4424 non-null   int64  
     6   Nacionality                                     4424 non-null   int64  
     7   Mother's qualification                          4424 non-null   int64  
     8   Father's qualification                          4424 non-null   int64  
     9   Mother's occupation                             4424 non-null   int64  
     10  Father's occupation                             4424 non-null   int64  
     11  Displaced                                       4424 non-null   int64  
     12  Educational special needs                       4424 non-null   int64  
     13  Debtor                                          4424 non-null   int64  
     14  Tuition fees up to date                         4424 non-null   int64  
     15  Gender                                          4424 non-null   int64  
     16  Scholarship holder                              4424 non-null   int64  
     17  Age at enrollment                               4424 non-null   int64  
     18  International                                   4424 non-null   int64  
     19  Curricular units 1st sem (credited)             4424 non-null   int64  
     20  Curricular units 1st sem (enrolled)             4424 non-null   int64  
     21  Curricular units 1st sem (evaluations)          4424 non-null   int64  
     22  Curricular units 1st sem (approved)             4424 non-null   int64  
     23  Curricular units 1st sem (grade)                4424 non-null   float64
     24  Curricular units 1st sem (without evaluations)  4424 non-null   int64  
     25  Curricular units 2nd sem (credited)             4424 non-null   int64  
     26  Curricular units 2nd sem (enrolled)             4424 non-null   int64  
     27  Curricular units 2nd sem (evaluations)          4424 non-null   int64  
     28  Curricular units 2nd sem (approved)             4424 non-null   int64  
     29  Curricular units 2nd sem (grade)                4424 non-null   float64
     30  Curricular units 2nd sem (without evaluations)  4424 non-null   int64  
     31  Unemployment rate                               4424 non-null   float64
     32  Inflation rate                                  4424 non-null   float64
     33  GDP                                             4424 non-null   float64
     34  Target                                          4424 non-null   object 
    dtypes: float64(5), int64(29), object(1)
    memory usage: 1.2+ MB
    

We have most of our data in numbers as intergers and floats however we object data as Target.We must consider this in following steps as pre-processing step.


```python
df.isna().sum()
### sanity check to make sure there's no null value in the columns 
```




    Marital status                                    0
    Application mode                                  0
    Application order                                 0
    Course                                            0
    Daytime/evening attendance                        0
    Previous qualification                            0
    Nacionality                                       0
    Mother's qualification                            0
    Father's qualification                            0
    Mother's occupation                               0
    Father's occupation                               0
    Displaced                                         0
    Educational special needs                         0
    Debtor                                            0
    Tuition fees up to date                           0
    Gender                                            0
    Scholarship holder                                0
    Age at enrollment                                 0
    International                                     0
    Curricular units 1st sem (credited)               0
    Curricular units 1st sem (enrolled)               0
    Curricular units 1st sem (evaluations)            0
    Curricular units 1st sem (approved)               0
    Curricular units 1st sem (grade)                  0
    Curricular units 1st sem (without evaluations)    0
    Curricular units 2nd sem (credited)               0
    Curricular units 2nd sem (enrolled)               0
    Curricular units 2nd sem (evaluations)            0
    Curricular units 2nd sem (approved)               0
    Curricular units 2nd sem (grade)                  0
    Curricular units 2nd sem (without evaluations)    0
    Unemployment rate                                 0
    Inflation rate                                    0
    GDP                                               0
    Target                                            0
    dtype: int64



We should also check are there any null values! All the values are filled with data.


```python
#Checking the shape of the data
df.shape
```




    (4424, 35)




```python
#Have an overview idea with a quick visualzation as pie chart
student_target = df['Target'].value_counts()
colors=sns.color_palette('Set2')
plt.pie(student_target, labels=student_target.index, autopct= '%1.1f%%',colors=colors)

plt.title('Percentage of Student Target')
plt.show()
```


    
![png](output_18_0.png)
    



```python
df['Target'].hist()
```




    <Axes: >




    
![png](output_19_1.png)
    



```python
sns.catplot(data=df, x='Target', kind='count')

```




    <seaborn.axisgrid.FacetGrid at 0x261bf57a740>




    
![png](output_20_1.png)
    


Regarding the target variable, almost half of the data falls within the Graduate category.




```python
#Alos checking the statistical data
df.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>...</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.178571</td>
      <td>6.886980</td>
      <td>1.727848</td>
      <td>9.899186</td>
      <td>0.890823</td>
      <td>2.531420</td>
      <td>1.254521</td>
      <td>12.322107</td>
      <td>16.455244</td>
      <td>7.317812</td>
      <td>...</td>
      <td>0.137658</td>
      <td>0.541817</td>
      <td>6.232143</td>
      <td>8.063291</td>
      <td>4.435805</td>
      <td>10.230206</td>
      <td>0.150316</td>
      <td>11.566139</td>
      <td>1.228029</td>
      <td>0.001969</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.605747</td>
      <td>5.298964</td>
      <td>1.313793</td>
      <td>4.331792</td>
      <td>0.311897</td>
      <td>3.963707</td>
      <td>1.748447</td>
      <td>9.026251</td>
      <td>11.044800</td>
      <td>3.997828</td>
      <td>...</td>
      <td>0.690880</td>
      <td>1.918546</td>
      <td>2.195951</td>
      <td>3.947951</td>
      <td>3.014764</td>
      <td>5.210808</td>
      <td>0.753774</td>
      <td>2.663850</td>
      <td>1.382711</td>
      <td>2.269935</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.600000</td>
      <td>-0.800000</td>
      <td>-4.060000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>10.750000</td>
      <td>0.000000</td>
      <td>9.400000</td>
      <td>0.300000</td>
      <td>-1.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>14.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>12.200000</td>
      <td>0.000000</td>
      <td>11.100000</td>
      <td>1.400000</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>22.000000</td>
      <td>27.000000</td>
      <td>10.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>13.333333</td>
      <td>0.000000</td>
      <td>13.900000</td>
      <td>2.600000</td>
      <td>1.790000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>18.000000</td>
      <td>9.000000</td>
      <td>17.000000</td>
      <td>1.000000</td>
      <td>17.000000</td>
      <td>21.000000</td>
      <td>29.000000</td>
      <td>34.000000</td>
      <td>32.000000</td>
      <td>...</td>
      <td>12.000000</td>
      <td>19.000000</td>
      <td>23.000000</td>
      <td>33.000000</td>
      <td>20.000000</td>
      <td>18.571429</td>
      <td>12.000000</td>
      <td>16.200000</td>
      <td>3.700000</td>
      <td>3.510000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>




```python
#  The data have object which means it is categorical data, now converting categorical data to numeric data.
dummies_df = pd.get_dummies(df)
display(df.head())
display(df.tail())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>10</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>28</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4419</th>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>5</td>
      <td>12.666667</td>
      <td>0</td>
      <td>15.5</td>
      <td>2.8</td>
      <td>-4.06</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4420</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>11.000000</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>4421</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>9</td>
      <td>1</td>
      <td>13.500000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>4422</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>12.000000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4423</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>12.7</td>
      <td>3.7</td>
      <td>-1.70</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>


Now they are still strings (Target values) not numbers 


```python
# Convert Yes to 1 and No to 0 for the Target:

df['Target'] = df['Target'].map({'Graduate': 0, 'Dropout': 1, 'Enrolled': 2 })
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>10</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>28</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



Now the Target got numerical valeus as 0,1,2.


```python
# Before the EDA, dropping the values would be benefical, those are the ones are not  affecting the  course and it's factors to determine a pass or not
#as folllows: Marital status, attendance, self/parent qualifications and occupations, special needs,gender, age, international, do not have an affect on Target

df = df.drop(["Application mode","Marital status", "Application order", "Daytime/evening attendance",
              "International","Nacionality", "Mother's qualification", "Father's qualification","Debtor","Mother's occupation",
              "Father's qualification","Displaced", "Educational special needs", "Gender",
              "Age at enrollment", ], axis=1)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Course</th>
      <th>Previous qualification</th>
      <th>Father's occupation</th>
      <th>Tuition fees up to date</th>
      <th>Scholarship holder</th>
      <th>Curricular units 1st sem (credited)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>14.000000</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>6</td>
      <td>13.428571</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>5</td>
      <td>12.333333</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.dtypes
```




    Course                                              int64
    Previous qualification                              int64
    Father's occupation                                 int64
    Tuition fees up to date                             int64
    Scholarship holder                                  int64
    Curricular units 1st sem (credited)                 int64
    Curricular units 1st sem (enrolled)                 int64
    Curricular units 1st sem (evaluations)              int64
    Curricular units 1st sem (approved)                 int64
    Curricular units 1st sem (grade)                  float64
    Curricular units 1st sem (without evaluations)      int64
    Curricular units 2nd sem (credited)                 int64
    Curricular units 2nd sem (enrolled)                 int64
    Curricular units 2nd sem (evaluations)              int64
    Curricular units 2nd sem (approved)                 int64
    Curricular units 2nd sem (grade)                  float64
    Curricular units 2nd sem (without evaluations)      int64
    Unemployment rate                                 float64
    Inflation rate                                    float64
    GDP                                               float64
    Target                                              int64
    dtype: object



Now we have all relevant data.


```python
df.shape
```




    (4424, 21)



It was 35 columns a-now the rows are decreased to 21.

Since our data is clean and complete we can do the EDA analysis.


```python
df.shape
df.count()
```




    Course                                            4424
    Previous qualification                            4424
    Father's occupation                               4424
    Tuition fees up to date                           4424
    Scholarship holder                                4424
    Curricular units 1st sem (credited)               4424
    Curricular units 1st sem (enrolled)               4424
    Curricular units 1st sem (evaluations)            4424
    Curricular units 1st sem (approved)               4424
    Curricular units 1st sem (grade)                  4424
    Curricular units 1st sem (without evaluations)    4424
    Curricular units 2nd sem (credited)               4424
    Curricular units 2nd sem (enrolled)               4424
    Curricular units 2nd sem (evaluations)            4424
    Curricular units 2nd sem (approved)               4424
    Curricular units 2nd sem (grade)                  4424
    Curricular units 2nd sem (without evaluations)    4424
    Unemployment rate                                 4424
    Inflation rate                                    4424
    GDP                                               4424
    Target                                            4424
    dtype: int64




```python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap between Variables')
plt.show()
```


    
![png](output_34_0.png)
    



```python

plt.figure(figsize=(15, 10))
sns.heatmap(data=df.corr(), annot=True)
```




    <Axes: >




    
![png](output_35_1.png)
    



```python
r = df.loc[:,["Target","Curricular units 2nd sem (grade)","Curricular units 1st sem (evaluations)","Curricular units 2nd sem (evaluations)","Curricular units 1st sem (enrolled)","Curricular units 2nd sem (enrolled)","Curricular units 1st sem (grade)","Scholarship holder","Curricular units 1st sem (approved)",]]
     
sns.heatmap(r.corr(),vmin = -1,vmax = 1,annot = True);
```


    
![png](output_36_0.png)
    



```python
sns.boxplot(x=df['Curricular units 1st sem (grade)'])
```




    <Axes: xlabel='Curricular units 1st sem (grade)'>




    
![png](output_37_1.png)
    



```python
sns.boxplot(x=df['Curricular units 2nd sem (grade)'])
```




    <Axes: xlabel='Curricular units 2nd sem (grade)'>




    
![png](output_38_1.png)
    


We concentrated on the grades of the each semester because this affects the graduation of the students .



```python
#We neglected the enrolled students
frame = df.loc[:,list(df.columns[list(df.columns.str.contains("grade|approved",regex = True))])]

```


```python

frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>14.000000</td>
      <td>6</td>
      <td>13.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>13.428571</td>
      <td>5</td>
      <td>12.400000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>12.333333</td>
      <td>6</td>
      <td>13.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

#Extracting important  columns for machine learning

important_cols = ["Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)","Curricular units 1st sem (grade)","Curricular units 1st sem (approved)","Tuition fees up to date" ,"Target"]
#data containing just critical columns
df = df[important_cols]

#data containing only graduates and dropouts we concentrated on  predicting graduates and dropouts.
df = df[df["Target"] != 2]
     
```


```python
frame = df[important_cols]
Q1 = frame.quantile(0.25)
Q3 = frame.quantile(0.75)
IQR = Q3 - Q1
ml_data = df[~((frame < (Q1 - 1.5 * IQR)) |(frame > (Q3 + 1.5 * IQR))).any(axis=1)]
```

Removing outlayers of the crucial data.


```python

corr_matrix=df.corr()
corr_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Tuition fees up to date</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Curricular units 2nd sem (approved)</th>
      <td>1.000000</td>
      <td>0.786838</td>
      <td>0.691907</td>
      <td>0.916334</td>
      <td>0.329017</td>
      <td>0.653995</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (grade)</th>
      <td>0.786838</td>
      <td>1.000000</td>
      <td>0.845864</td>
      <td>0.709368</td>
      <td>0.318721</td>
      <td>0.605350</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (grade)</th>
      <td>0.691907</td>
      <td>0.845864</td>
      <td>1.000000</td>
      <td>0.710157</td>
      <td>0.275555</td>
      <td>0.519927</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (approved)</th>
      <td>0.916334</td>
      <td>0.709368</td>
      <td>0.710157</td>
      <td>1.000000</td>
      <td>0.277787</td>
      <td>0.554881</td>
    </tr>
    <tr>
      <th>Tuition fees up to date</th>
      <td>0.329017</td>
      <td>0.318721</td>
      <td>0.275555</td>
      <td>0.277787</td>
      <td>1.000000</td>
      <td>0.442138</td>
    </tr>
    <tr>
      <th>Target</th>
      <td>0.653995</td>
      <td>0.605350</td>
      <td>0.519927</td>
      <td>0.554881</td>
      <td>0.442138</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Project 4

Reagrding the regression models we must seperate the data as y and x variables before this step we first evaluate the data again.

## Splitting Data


```python
# Reloading the data set
df = pd.read_csv("Student_academic.csv")
df = df.drop(df[df['Target']=='Enrolled'].index)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>Mother's occupation</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>10</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>27</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>27</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>28</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
#Again the useless columns are dropped out for better model evaluation.

df = df.drop(["Marital status", "Application mode", "Application order", "Daytime/evening attendance",
              "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
              "Father's occupation", "Displaced", "Educational special needs", "Debtor", "Gender",
              "Age at enrollment", "International"], axis=1)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Course</th>
      <th>Previous qualification</th>
      <th>Tuition fees up to date</th>
      <th>Scholarship holder</th>
      <th>Curricular units 1st sem (credited)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>14.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>6</td>
      <td>13.428571</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>5</td>
      <td>12.333333</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Separate the data into labels and features
# Separate the y variable, the labels
Y = df['Target']

# Separate the X variable, the features
X = df.drop(columns=['Target'])
```


```python
encoder = LabelEncoder()
```


```python
df['Target'] = encoder.fit_transform(df['Target'])
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Course</th>
      <th>Previous qualification</th>
      <th>Tuition fees up to date</th>
      <th>Scholarship holder</th>
      <th>Curricular units 1st sem (credited)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>14.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>6</td>
      <td>13.428571</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>5</td>
      <td>12.333333</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df.drop(columns=['Target'], axis=1)
Y = df['Target']
```


```python
print(X, X.shape)
```

          Course  Previous qualification  Tuition fees up to date  \
    0          2                       1                        1   
    1         11                       1                        0   
    2          5                       1                        0   
    3         15                       1                        1   
    4          3                       1                        1   
    ...      ...                     ...                      ...   
    4419      15                       1                        1   
    4420      15                       1                        0   
    4421      12                       1                        1   
    4422       9                       1                        1   
    4423      15                       1                        1   
    
          Scholarship holder  Curricular units 1st sem (credited)  \
    0                      0                                    0   
    1                      0                                    0   
    2                      0                                    0   
    3                      0                                    0   
    4                      0                                    0   
    ...                  ...                                  ...   
    4419                   0                                    0   
    4420                   0                                    0   
    4421                   1                                    0   
    4422                   1                                    0   
    4423                   0                                    0   
    
          Curricular units 1st sem (enrolled)  \
    0                                       0   
    1                                       6   
    2                                       6   
    3                                       6   
    4                                       6   
    ...                                   ...   
    4419                                    6   
    4420                                    6   
    4421                                    7   
    4422                                    5   
    4423                                    6   
    
          Curricular units 1st sem (evaluations)  \
    0                                          0   
    1                                          6   
    2                                          0   
    3                                          8   
    4                                          9   
    ...                                      ...   
    4419                                       7   
    4420                                       6   
    4421                                       8   
    4422                                       5   
    4423                                       8   
    
          Curricular units 1st sem (approved)  Curricular units 1st sem (grade)  \
    0                                       0                          0.000000   
    1                                       6                         14.000000   
    2                                       0                          0.000000   
    3                                       6                         13.428571   
    4                                       5                         12.333333   
    ...                                   ...                               ...   
    4419                                    5                         13.600000   
    4420                                    6                         12.000000   
    4421                                    7                         14.912500   
    4422                                    5                         13.800000   
    4423                                    6                         11.666667   
    
          Curricular units 1st sem (without evaluations)  \
    0                                                  0   
    1                                                  0   
    2                                                  0   
    3                                                  0   
    4                                                  0   
    ...                                              ...   
    4419                                               0   
    4420                                               0   
    4421                                               0   
    4422                                               0   
    4423                                               0   
    
          Curricular units 2nd sem (credited)  \
    0                                       0   
    1                                       0   
    2                                       0   
    3                                       0   
    4                                       0   
    ...                                   ...   
    4419                                    0   
    4420                                    0   
    4421                                    0   
    4422                                    0   
    4423                                    0   
    
          Curricular units 2nd sem (enrolled)  \
    0                                       0   
    1                                       6   
    2                                       6   
    3                                       6   
    4                                       6   
    ...                                   ...   
    4419                                    6   
    4420                                    6   
    4421                                    8   
    4422                                    5   
    4423                                    6   
    
          Curricular units 2nd sem (evaluations)  \
    0                                          0   
    1                                          6   
    2                                          0   
    3                                         10   
    4                                          6   
    ...                                      ...   
    4419                                       8   
    4420                                       6   
    4421                                       9   
    4422                                       6   
    4423                                       6   
    
          Curricular units 2nd sem (approved)  Curricular units 2nd sem (grade)  \
    0                                       0                          0.000000   
    1                                       6                         13.666667   
    2                                       0                          0.000000   
    3                                       5                         12.400000   
    4                                       6                         13.000000   
    ...                                   ...                               ...   
    4419                                    5                         12.666667   
    4420                                    2                         11.000000   
    4421                                    1                         13.500000   
    4422                                    5                         12.000000   
    4423                                    6                         13.000000   
    
          Curricular units 2nd sem (without evaluations)  Unemployment rate  \
    0                                                  0               10.8   
    1                                                  0               13.9   
    2                                                  0               10.8   
    3                                                  0                9.4   
    4                                                  0               13.9   
    ...                                              ...                ...   
    4419                                               0               15.5   
    4420                                               0               11.1   
    4421                                               0               13.9   
    4422                                               0                9.4   
    4423                                               0               12.7   
    
          Inflation rate   GDP  
    0                1.4  1.74  
    1               -0.3  0.79  
    2                1.4  1.74  
    3               -0.8 -3.12  
    4               -0.3  0.79  
    ...              ...   ...  
    4419             2.8 -4.06  
    4420             0.6  2.02  
    4421            -0.3  0.79  
    4422            -0.8 -3.12  
    4423             3.7 -1.70  
    
    [3630 rows x 19 columns] (3630, 19)
    

Now we are eliminating the enrolled students which are not be evaluated.

## Model training



```python
from sklearn.model_selection import train_test_split 
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```


```python

X.shape, Y.shape
```




    ((3630, 19), (3630,))




```python
standard_scaler = StandardScaler()
X_scaled_train = standard_scaler.fit_transform(X_train)
X_scaled_test = standard_scaler.fit_transform(X_test)

pca = PCA(n_components=3)
X_scaled_train= pca.fit_transform(X_scaled_train)
X_scaled_test = pca.transform(X_scaled_test)
```


```python
X_train.shape,Y_train.shape
```




    ((2904, 19), (2904,))




```python
X_test.shape, Y_test.shape
```




    ((726, 19), (726,))



80% of the 3630 rows an 726 is the 20 % of it.


```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
```


```python
lm.fit(X_train,Y_train)

```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
Y_pred=lm.predict(X_test)
```


```python
print('Coefficients:', lm.coef_)
print('Intercept:', lm.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))
```

    Coefficients: [-0.00765433  0.00229119  0.20882785  0.07917694 -0.03503912  0.00578308
     -0.00413174  0.04633724 -0.00300596  0.0236162  -0.03148679 -0.07414407
     -0.01084351  0.12756268 -0.00563561  0.00597957 -0.00201112  0.00351422
      0.00283408]
    Intercept: 0.3749918316234002
    Mean squared error (MSE): 0.09
    Coefficient of determination (R^2): 0.64
    


```python
r2_score(Y_test, Y_pred)
```




    0.640767502245412



From Linear Regrression we can evaluate the line graph eqaution of Y pred value with is a preicted value of Y (Target values), these coeiffents used to make the equation for y and the intercept value is also given.From the linear regression the R^2 value is the key parameter for our model, which is 0.64. This model is not a super sucessful model,From the Linear reg we know that the  R^2 is getting close to the 1 is a desired goal for measuring the sucess of the model and  the score value is 0.64.


```python

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("RMSE value: {:.4f}".format(rmse))
```

    RMSE value: 0.2945
    


```python
print ("R2 Score value: {:.4f}".format(r2_score(Y_test, Y_pred)))
```

    R2 Score value: 0.6408
    

The RMSE value has been found to be 0.2605. It means the standard deviation for our prediction is 2.94, which is not a huge error.
In business decisions, the benchmark for the R2 score value is 0.7. It means if R2 score value <= 0.7, then the model is not  good enough to deploy. Our R2 score value has been found to be 0.6408. It means that this model explains 64 % of the variance in our dependent variable.Despite we have small RMSE, the R2 score value is not confirms that the model is  good enough to deploy.

Now we can try other models for better results


```python
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
```

    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, Y_train)

svm = SVC()
svm.fit(X_train, Y_train)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">GaussianNB</label><div class="sk-toggleable__content"><pre>GaussianNB()</pre></div></div></div></div></div>




```python
logreg_predictions = logreg.predict(X_test)
random_forest_predictions = random_forest.predict(X_test)
decision_tree_predictions = decision_tree.predict(X_test)
svm_predictions = svm.predict(X_test)
naive_bayes_predictions = naive_bayes.predict(X_test)
```


```python
logreg_accuracy = accuracy_score(Y_test, logreg_predictions)
random_forest_accuracy = accuracy_score(Y_test, random_forest_predictions)
decision_tree_accuracy = accuracy_score(Y_test, decision_tree_predictions)
svm_accuracy = accuracy_score(Y_test, svm_predictions)
naive_bayes_accuracy = accuracy_score(Y_test, naive_bayes_predictions)

print("Logistic Regression Accuracy:", logreg_accuracy)
print("Random Forest Accuracy:", random_forest_accuracy)
print("Decision Tree Accuracy:", decision_tree_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Naive Bayes Accuracy:", naive_bayes_accuracy)
```

    Logistic Regression Accuracy: 0.9035812672176309
    Random Forest Accuracy: 0.8980716253443526
    Decision Tree Accuracy: 0.8415977961432507
    SVM Accuracy: 0.8856749311294766
    Naive Bayes Accuracy: 0.8526170798898072
    

Regarding the accuracy the best models for our data is Logistic Regression> Random Forest> Support Vector Machine (SVM) > Naive Bayes > Decision Tree.

## Checking for Overfitting and Underfitting


```python
print("Training set score: {:.4f}".format(lm.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(lm.score(X_test,Y_test)))
```

    Training set score: 0.6576
    Test set score: 0.6408
    


```python
print("Training set score: {:.4f}".format(logreg.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(logreg.score(X_test,Y_test)))
```

    Training set score: 0.9122
    Test set score: 0.9036
    


```python
print("Training set score: {:.4f}".format(random_forest.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(random_forest.score(X_test,Y_test)))
```

    Training set score: 0.9910
    Test set score: 0.8981
    


```python
print("Training set score: {:.4f}".format(decision_tree.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(decision_tree.score(X_test,Y_test)))
```

    Training set score: 0.9910
    Test set score: 0.8416
    


```python
print("Training set score: {:.4f}".format(svm.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(svm.score(X_test,Y_test)))
```

    Training set score: 0.9012
    Test set score: 0.8857
    


```python
print("Training set score: {:.4f}".format(naive_bayes.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(naive_bayes.score(X_test,Y_test)))
```

    Training set score: 0.8502
    Test set score: 0.8526
    

The training/test set scores are important for evaluating performance of the models.
Test score and training score are desired to be closer values. 
If the scores are close values that means the model performs well on the training data.In other words, models learn the relationships appropriately from the training data.

Regrading to our models "Random Forest" and "Decision Tree" have the highest scores among all.However, we can see higher difference between the training and test scores.They would be not the perfect fit.

Okay now we can do predictions based on these 5 models:

## Prediction with Models


```python
#Re-uplading the csv file with Target values as numeric and relevant data in frame.
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Course</th>
      <th>Previous qualification</th>
      <th>Tuition fees up to date</th>
      <th>Scholarship holder</th>
      <th>Curricular units 1st sem (credited)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>14.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>6</td>
      <td>13.428571</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>5</td>
      <td>12.333333</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Okay from our data frame we take the first row valeus for prediction of the Target value as "Dropout",which is Target:0 
#lets see.

input_data = (2, 1, 1, 0, 0, 0, 0, 0, 0.000000, 0, 0, 0, 0, 0, 0.000000, 0, 10.8, 1.4, 1.74)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction_logreg = logreg.predict(input_data_reshaped)
prediction_rf = random_forest.predict(input_data_reshaped)
prediction_dt = decision_tree.predict(input_data_reshaped)
prediction_svm = svm.predict(input_data_reshaped)
prediction_nb = naive_bayes.predict(input_data_reshaped)

print("Logistic Regression Prediction:", prediction_logreg[0])
print("Random Forest Prediction:", prediction_rf[0])
print("Decision Tree Prediction:", prediction_dt[0])
print("SVM Prediction:", prediction_svm[0])
print("Naive Bayes Prediction:", prediction_nb[0])

```

    Logistic Regression Prediction: 1
    Random Forest Prediction: 0
    Decision Tree Prediction: 0
    SVM Prediction: 0
    Naive Bayes Prediction: 0
    

    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but SVC was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but GaussianNB was fitted with feature names
      warnings.warn(
    

Now we have the outputs of the Target of the first row so basically at first  row we have the Target value of 0  meaning that  stduent is "Dropout" the degree remembering that we have only 0s and 1s as Dropout:0, Gardaute:1.Unlike the accuracy Logistic Regression failed to predict the dropout target value among all the models.


```python
#Now we can predict with the all models  for Graduate students as Target:1
#Taking the second row as a example(Using numpy array reshape function data frame to array)

input_data = (11,1,0,0,0,6,6,6,14.000000,0,0,6,6,6, 13.666667, 0, 13.9, -0.3, 0.79)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction_logreg = logreg.predict(input_data_reshaped)
prediction_rf = random_forest.predict(input_data_reshaped)
prediction_dt = decision_tree.predict(input_data_reshaped)
prediction_svm = svm.predict(input_data_reshaped)
prediction_nb = naive_bayes.predict(input_data_reshaped)

print("Logistic Regression Prediction:", prediction_logreg[0])
print("Random Forest Prediction:", prediction_rf[0])
print("Decision Tree Prediction:", prediction_dt[0])
print("LinearSVC:", prediction_svm[0])
print("Naive Bayes Prediction:", prediction_nb[0])
```

    Logistic Regression Prediction: 1
    Random Forest Prediction: 1
    Decision Tree Prediction: 1
    LinearSVC: 1
    Naive Bayes Prediction: 0
    

    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but SVC was fitted with feature names
      warnings.warn(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but GaussianNB was fitted with feature names
      warnings.warn(
    

Now we can see that only Naive Bayes predicted wrongly the graduate students as a model.
We have two models having issues with predictions before it was Logistic regrression now is Naive Bayes.
We make the model more complex with Logistic regression since we have binary data as 0s and 1s.Moreover, our model is also a catergorical model we can work with also categorical models.


## Checking Performance Metrics


```python
from xgboost import XGBClassifier

models = {"Logistic Regression":LogisticRegression(),
           "SVC":LinearSVC(),
          "XGBoost":XGBClassifier(),
          "Random Forest": RandomForestClassifier(), 
          "Decision Tree": DecisionTreeClassifier(),
          "Naives Bayes": GaussianNB()}


log = LogisticRegression()
log.fit(X_train,Y_train)

rndf =  RandomForestClassifier()
rndf.fit(X_train,Y_train)

lsvc = LinearSVC()
lsvc.fit(X_train,Y_train)

xgb = XGBClassifier()
xgb.fit(X_train,Y_train)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,Y_train)

gnb = GaussianNB()
gnb.fit(X_train,Y_train)

```

    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\svm\_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(
    




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">GaussianNB</label><div class="sk-toggleable__content"><pre>GaussianNB()</pre></div></div></div></div></div>




```python
arr = []

for name,model in models.items():
  current_model = model.fit(X_train,Y_train)
  Y_pred = model.predict(X_test)
  print(name)
  print("Accuracy:", "%.3f" % metrics.accuracy_score(Y_test, Y_pred))
  print("Precision:", "%.3f" % metrics.precision_score(Y_test, Y_pred))
  print("Recall:", "%.3f" % metrics.recall_score(Y_test, Y_pred))
  print("F1 Score:", "%.3f" % metrics.f1_score(Y_test, Y_pred))
  arr.append([name,"%.3f" % metrics.accuracy_score(Y_test, Y_pred),
              "%.3f" % metrics.precision_score(Y_test, Y_pred),
              "%.3f" % metrics.recall_score(Y_test, Y_pred),
              "%.3f" % metrics.f1_score(Y_test, Y_pred)])
  print("\n")
```

    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Logistic Regression
    Accuracy: 0.904
    Precision: 0.891
    Recall: 0.953
    F1 Score: 0.921
    
    
    

    C:\Users\MELDA\anaconda3\lib\site-packages\sklearn\svm\_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(
    

    SVC
    Accuracy: 0.905
    Precision: 0.882
    Recall: 0.970
    F1 Score: 0.924
    
    
    XGBoost
    Accuracy: 0.897
    Precision: 0.894
    Recall: 0.937
    F1 Score: 0.915
    
    
    Random Forest
    Accuracy: 0.895
    Precision: 0.881
    Recall: 0.951
    F1 Score: 0.915
    
    
    Decision Tree
    Accuracy: 0.840
    Precision: 0.862
    Recall: 0.870
    F1 Score: 0.866
    
    
    Naives Bayes
    Accuracy: 0.853
    Precision: 0.836
    Recall: 0.935
    F1 Score: 0.883
    
    
    


```python
pd.DataFrame({"Model":list(models.keys()),"Accuracy":[i[1] for i in arr],"Precision":[i[2] for i in arr],"Recall":[i[3] for i in arr],"F1 score":[i[4] for i in arr]})
     
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.904</td>
      <td>0.891</td>
      <td>0.953</td>
      <td>0.921</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVC</td>
      <td>0.902</td>
      <td>0.886</td>
      <td>0.958</td>
      <td>0.921</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>0.897</td>
      <td>0.894</td>
      <td>0.937</td>
      <td>0.915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>0.888</td>
      <td>0.882</td>
      <td>0.937</td>
      <td>0.909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decision Tree</td>
      <td>0.851</td>
      <td>0.869</td>
      <td>0.881</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Naives Bayes</td>
      <td>0.853</td>
      <td>0.836</td>
      <td>0.935</td>
      <td>0.883</td>
    </tr>
  </tbody>
</table>
</div>



"Precision","Recall" and "F1 Score" is the total overall accuracy measure of the classification model. We should take into account that F1 score, but you must also consider recall.Since we have binary classsifcation it is enough just to consider accuracy and F1.We must take into account in some risky (e.g.health and justice system data) cases we cannot avoid Recall for avoiding true negative results (TYPE II ERRORS).
Also we can use ROC-AUC graphs since we have binary classification.Also we can check the confusion matrix.Confusion matrix is known as how many times the model predict the class.


```python
from sklearn.metrics import roc_curve, roc_auc_score

```


```python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
algos = [log,rndf,lsvc,xgb,dtree,gnb]
cnt = 0
colour = ["b","g","k","r","c","m"]

for i in range(0,2):
  for j in range(0,3):
    Y_pred_proba = algos[cnt].predict_proba(X_test)[:,-1] if cnt != 2 else algos[cnt].decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
    auc = roc_auc_score(Y_test, Y_pred_proba)
    axes[i,j].plot(fpr, tpr, label = "AUC ROC: {:.2f}".format(auc),color = colour[cnt])
    axes[i,j].set_xlabel('False Positive Rate')
    axes[i,j].set_ylabel('True Positive Rate')
    axes[i,j].legend(loc = "lower right")
    axes[i,j].set_title("{} Regression AUC ROC".format(list(models.keys())[cnt]))
    cnt+=1
plt.subplots_adjust(hspace=0.4, wspace=0.8)

```


    
![png](output_101_0.png)
    


From the ROC Curves area under the curves AUC and they are similiarly good.Obviously, Logistic Regression and XGBoost took the the lead with 0.95 and Support Vector Machines and Random Forest have the AUC of 0.94 which is also good result.


```python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
cnt = 0
for i in range(0,2):
  for j in range(0,3):
    model_pred = algos[cnt].predict(X_test)
    conf_m = confusion_matrix(Y_test, model_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_m, display_labels=['Dropout', 'Graduate'],)
    display.plot(values_format='',ax = axes[i,j])
    display.ax_.set_title("{} confusion matrix".format(list(models.keys())[cnt]))
    cnt+=1
  plt.tight_layout()

```


    
![png](output_103_0.png)
    


Logistic Regression has the best classifier model because it has "True Positive" prediction.

## Model Evaluation


```python
Training_score = [round(i.score(X_train, Y_train) * 100, 2) for i in algos]
Testing_score = [round(i.score(X_test, Y_test) * 100, 2) for i in algos]
     
```


```python
results_training = pd.DataFrame({"ML_Model":list(models.keys()),"Training score":Training_score})
results_testing = pd.DataFrame({"ML_Model":list(models.keys()),"Testing score":Testing_score})
```


```python

print(results_training)
print(results_testing)
```

                  ML_Model  Training score
    0  Logistic Regression           91.22
    1                  SVC           99.10
    2              XGBoost           90.63
    3        Random Forest           98.86
    4        Decision Tree           99.10
    5         Naives Bayes           85.02
                  ML_Model  Testing score
    0  Logistic Regression          90.36
    1                  SVC          90.08
    2              XGBoost          89.81
    3        Random Forest          89.67
    4        Decision Tree          83.61
    5         Naives Bayes          85.26
    

Training scores is higher than tesing scores which is a sign of over-fitting.It is due to the nature of the models.

Decision trees and random forest,and XGBoost, are categorical models that posses high flexiblity meaning that has high variance, we can tune in hyperparameters.

"Logistic Regression" performs best among all moedels for the university to predict the enrolled student will graduate or dropout.



```python

```
