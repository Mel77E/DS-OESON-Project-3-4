#!/usr/bin/env python
# coding: utf-8

# # OESON PROJECT 3&4  Graduation Prediction
# 

# ## Project Description

# This project involves the exploration and analysis and prediction of student data obtained from a reputed university that include relevant information of the students currently enrolled in  the university for a specific program like personal information and curriculum data.
# 
# As a Data Scientist I performed EDA on the data and suggested a prediction model which predicts whether a college student will graduate or not using at least six machine learning algorithms.
# 

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
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


# We need to dowload all necesary libraries for analysis and modeling of the data.

# For reading our data set as csv file we used pandas library.

# In[8]:


df = pd.read_csv("Student_academic.csv")
df


# We can see the whole dataset with all details of the students the data file has 4424 rows and 35 columns .
# With the help of head and tail commends we can have an idea about the data which we are concentrating on "Target".

# In[10]:


df = pd.read_csv("Student_academic.csv")
df.head(5)


# In[12]:


df.tail(5)


# Okay, now we want to learn more about our data types are there any objects or all of them are numerical values?
# 
# Exploratory Data Analysis (EDA) it is crutial to have the idea what kind of data we are working for, then we are able to do feature engineering accordinly, after all we can then go to the modelling part.

# ## Exploratory Data Analysis (EDA) 

# In[14]:


#Checking the column information of the data frame

df.dtypes


# In[326]:


#Checking the null count of the data an the number of data.
df.info()


# We have most of our data in numbers as intergers and floats however we object data as Target.We must consider this in following steps as pre-processing step.

# In[16]:


df.isna().sum()
### sanity check to make sure there's no null value in the columns 


# We should also check are there any null values! All the values are filled with data.

# In[18]:


#Checking the shape of the data
df.shape


# In[329]:


#Have an overview idea with a quick visualzation as pie chart
student_target = df['Target'].value_counts()
colors=sns.color_palette('Set2')
plt.pie(student_target, labels=student_target.index, autopct= '%1.1f%%',colors=colors)

plt.title('Percentage of Student Target')
plt.show()


# In[330]:


df['Target'].hist()


# In[331]:


sns.catplot(data=df, x='Target', kind='count')


# Regarding the target variable, almost half of the data falls within the Graduate category.
# 
# 

# In[20]:


#Alos checking the statistical data
df.describe()


# In[22]:


#  The data have object which means it is categorical data, now converting categorical data to numeric data.
dummies_df = pd.get_dummies(df)
display(df.head())
display(df.tail())


# Now they are still strings (Target values) not numbers 

# In[24]:


# Convert Yes to 1 and No to 0 for the Target:

df['Target'] = df['Target'].map({'Graduate': 0, 'Dropout': 1, 'Enrolled': 2 })
df.head()


# Now the Target got numerical valeus as 0,1,2.

# In[26]:


# Before the EDA, dropping the values would be benefical, those are the ones are not  affecting the  course and it's factors to determine a pass or not
#as folllows: Marital status, attendance, self/parent qualifications and occupations, special needs,gender, age, international, do not have an affect on Target

df = df.drop(["Application mode","Marital status", "Application order", "Daytime/evening attendance",
              "International","Nacionality", "Mother's qualification", "Father's qualification","Debtor","Mother's occupation",
              "Father's qualification","Displaced", "Educational special needs", "Gender",
              "Age at enrollment", ], axis=1)
df.head(5)


# In[163]:


df.dtypes


# Now we have all relevant data.

# In[164]:


df.shape


# It was 35 columns a-now the rows are decreased to 21.

# Since our data is clean and complete we can do the EDA analysis.

# In[28]:


df.shape
df.count()


# In[166]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap between Variables')
plt.show()


# In[257]:


plt.figure(figsize=(15, 10))
sns.heatmap(data=df.corr(), annot=True)


# In[258]:


r = df.loc[:,["Target","Curricular units 2nd sem (grade)","Curricular units 1st sem (evaluations)","Curricular units 2nd sem (evaluations)","Curricular units 1st sem (enrolled)","Curricular units 2nd sem (enrolled)","Curricular units 1st sem (grade)","Scholarship holder","Curricular units 1st sem (approved)",]]
     
sns.heatmap(r.corr(),vmin = -1,vmax = 1,annot = True);


# In[167]:


sns.boxplot(x=df['Curricular units 1st sem (grade)'])


# In[169]:


sns.boxplot(x=df['Curricular units 2nd sem (grade)'])


# We concentrated on the grades of the each semester because this affects the graduation of the students .
# 

# In[318]:


#We neglected the enrolled students
frame = df.loc[:,list(df.columns[list(df.columns.str.contains("grade|approved",regex = True))])]


# In[261]:


frame.head()


# In[272]:


#Extracting important  columns for machine learning

important_cols = ["Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)","Curricular units 1st sem (grade)","Curricular units 1st sem (approved)","Tuition fees up to date" ,"Target"]
#data containing just critical columns
df = df[important_cols]

#data containing only graduates and dropouts we concentrated on  predicting graduates and dropouts.
df = df[df["Target"] != 2]
     


# In[271]:


frame = df[important_cols]
Q1 = frame.quantile(0.25)
Q3 = frame.quantile(0.75)
IQR = Q3 - Q1
ml_data = df[~((frame < (Q1 - 1.5 * IQR)) |(frame > (Q3 + 1.5 * IQR))).any(axis=1)]


# Removing outlayers of the crucial data.

# In[273]:


corr_matrix=df.corr()
corr_matrix


# # Project 4

# Reagrding the regression models we must seperate the data as y and x variables before this step we first evaluate the data again.

# ## Splitting Data

# In[49]:


# Reloading the data set
df = pd.read_csv("Student_academic.csv")
df = df.drop(df[df['Target']=='Enrolled'].index)
df.head(5)


# In[51]:


#Again the useless columns are dropped out for better model evaluation.

df = df.drop(["Marital status", "Application mode", "Application order", "Daytime/evening attendance",
              "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
              "Father's occupation", "Displaced", "Educational special needs", "Debtor", "Gender",
              "Age at enrollment", "International"], axis=1)
df.head(5)


# In[53]:


# Separate the data into labels and features
# Separate the y variable, the labels
Y = df['Target']

# Separate the X variable, the features
X = df.drop(columns=['Target'])


# In[55]:


encoder = LabelEncoder()


# In[57]:


df['Target'] = encoder.fit_transform(df['Target'])
df.head(5)


# In[59]:


X = df.drop(columns=['Target'], axis=1)
Y = df['Target']


# In[61]:


print(X, X.shape)


# Now we are eliminating the enrolled students which are not be evaluated.

# ## Model training
# 

# In[63]:


from sklearn.model_selection import train_test_split 


# In[65]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[67]:


X.shape, Y.shape


# In[69]:


standard_scaler = StandardScaler()
X_scaled_train = standard_scaler.fit_transform(X_train)
X_scaled_test = standard_scaler.fit_transform(X_test)

pca = PCA(n_components=3)
X_scaled_train= pca.fit_transform(X_scaled_train)
X_scaled_test = pca.transform(X_scaled_test)


# In[71]:


X_train.shape,Y_train.shape


# In[73]:


X_test.shape, Y_test.shape


# 80% of the 3630 rows an 726 is the 20 % of it.

# In[75]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[77]:


lm.fit(X_train,Y_train)


# In[81]:


Y_pred=lm.predict(X_test)


# In[83]:


print('Coefficients:', lm.coef_)
print('Intercept:', lm.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))


# In[85]:


r2_score(Y_test, Y_pred)


# From Linear Regrression we can evaluate the line graph eqaution of Y pred value with is a preicted value of Y (Target values), these coeiffents used to make the equation for y and the intercept value is also given.From the linear regression the R^2 value is the key parameter for our model, which is 0.64. This model is not a super sucessful model,From the Linear reg we know that the  R^2 is getting close to the 1 is a desired goal for measuring the sucess of the model and  the score value is 0.64.

# In[87]:


mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("RMSE value: {:.4f}".format(rmse))


# In[291]:


print ("R2 Score value: {:.4f}".format(r2_score(Y_test, Y_pred)))


# The RMSE value has been found to be 0.2605. It means the standard deviation for our prediction is 2.94, which is not a huge error.
# In business decisions, the benchmark for the R2 score value is 0.7. It means if R2 score value <= 0.7, then the model is not  good enough to deploy. Our R2 score value has been found to be 0.6408. It means that this model explains 64 % of the variance in our dependent variable.Despite we have small RMSE, the R2 score value is not confirms that the model is  good enough to deploy.

# Now we can try other models for better results

# In[89]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


# In[91]:


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


# In[93]:


logreg_predictions = logreg.predict(X_test)
random_forest_predictions = random_forest.predict(X_test)
decision_tree_predictions = decision_tree.predict(X_test)
svm_predictions = svm.predict(X_test)
naive_bayes_predictions = naive_bayes.predict(X_test)


# In[95]:


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


# Regarding the accuracy the best models for our data is Logistic Regression> Random Forest> Support Vector Machine (SVM) > Naive Bayes > Decision Tree.

# ## Checking for Overfitting and Underfitting

# In[97]:


print("Training set score: {:.4f}".format(lm.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(lm.score(X_test,Y_test)))


# In[99]:


print("Training set score: {:.4f}".format(logreg.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(logreg.score(X_test,Y_test)))


# In[101]:


print("Training set score: {:.4f}".format(random_forest.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(random_forest.score(X_test,Y_test)))


# In[103]:


print("Training set score: {:.4f}".format(decision_tree.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(decision_tree.score(X_test,Y_test)))


# In[105]:


print("Training set score: {:.4f}".format(svm.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(svm.score(X_test,Y_test)))


# In[107]:


print("Training set score: {:.4f}".format(naive_bayes.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(naive_bayes.score(X_test,Y_test)))


# The training/test set scores are important for evaluating performance of the models.
# Test score and training score are desired to be closer values. 
# If the scores are close values that means the model performs well on the training data.In other words, models learn the relationships appropriately from the training data.
# 
# Regrading to our models "Random Forest" and "Decision Tree" have the highest scores among all.However, we can see higher difference between the training and test scores.They would be not the perfect fit.

# Okay now we can do predictions based on these 5 models:

# ## Prediction with Models

# In[119]:


#Re-uplading the csv file with Target values as numeric and relevant data in frame.
df.head(5)


# In[121]:


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


# Now we have the outputs of the Target of the first row so basically at first  row we have the Target value of 0  meaning that  stduent is "Dropout" the degree remembering that we have only 0s and 1s as Dropout:0, Gardaute:1.Unlike the accuracy Logistic Regression failed to predict the dropout target value among all the models.

# In[124]:


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


# Now we can see that only Naive Bayes predicted wrongly the graduate students as a model.
# We have two models having issues with predictions before it was Logistic regrression now is Naive Bayes.
# We make the model more complex with Logistic regression since we have binary data as 0s and 1s.Moreover, our model is also a catergorical model we can work with also categorical models.
# 

# ## Checking Performance Metrics

# In[127]:


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


# In[153]:


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


# In[131]:


pd.DataFrame({"Model":list(models.keys()),"Accuracy":[i[1] for i in arr],"Precision":[i[2] for i in arr],"Recall":[i[3] for i in arr],"F1 score":[i[4] for i in arr]})
     


# "Precision","Recall" and "F1 Score" is the total overall accuracy measure of the classification model. We should take into account that F1 score, but you must also consider recall.Since we have binary classsifcation it is enough just to consider accuracy and F1.We must take into account in some risky (e.g.health and justice system data) cases we cannot avoid Recall for avoiding true negative results (TYPE II ERRORS).
# Also we can use ROC-AUC graphs since we have binary classification.Also we can check the confusion matrix.Confusion matrix is known as how many times the model predict the class.

# In[134]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[138]:


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


# From the ROC Curves area under the curves AUC and they are similiarly good.Obviously, Logistic Regression and XGBoost took the the lead with 0.95 and Support Vector Machines and Random Forest have the AUC of 0.94 which is also good result.

# In[143]:


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


# Logistic Regression has the best classifier model because it has "True Positive" prediction.

# ## Model Evaluation

# In[145]:


Training_score = [round(i.score(X_train, Y_train) * 100, 2) for i in algos]
Testing_score = [round(i.score(X_test, Y_test) * 100, 2) for i in algos]
     


# In[147]:


results_training = pd.DataFrame({"ML_Model":list(models.keys()),"Training score":Training_score})
results_testing = pd.DataFrame({"ML_Model":list(models.keys()),"Testing score":Testing_score})


# In[149]:


print(results_training)
print(results_testing)


# Training scores is higher than tesing scores which is a sign of over-fitting.It is due to the nature of the models.
# 
# Decision trees and random forest,and XGBoost, are categorical models that posses high flexiblity meaning that has high variance, we can tune in hyperparameters.
# 
# "Logistic Regression" performs best among all moedels for the university to predict the enrolled student will graduate or dropout.
# 

# In[ ]:




