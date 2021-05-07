#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# get_ipython().system('python --version')
# 
# =============================================================================
#Context
#According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
#This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.
#
#Attribute Information
#1) id: unique identifier
#2) gender: "Male", "Female" or "Other"
#3) age: age of the patient
#4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
#5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
#6) ever_married: "No" or "Yes"
#7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
#8) Residence_type: "Rural" or "Urban"
#9) avg_glucose_level: average glucose level in blood
#10) bmi: body mass index
#11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
#12) stroke: 1 if the patient had a stroke or 0 if not
#*Note: "Unknown" in smoking_status means that the information is unavailable for this patientA stroke occurs when part of the brain loses its blood supply and stops working. This causes the part of the body that the injured brain controls to stop working. A stroke also is called a cerebrovascular accident, CVA, or "brain attack."
#
#The types of strokes include:
#
#Ischemic stroke (part of the brain loses blood flow)
#
#Hemorrhagic stroke (bleeding occurs within the brain)
#
#Transient ischemic attack, TIA, or mini-stroke (The stroke symptoms resolve within minutes, but may take up to 24 hours on their own without treatment. This is a warning sign that a stroke may occur in the near future.)
# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
import warnings
warnings.filterwarnings('ignore')
from sklearn import model_selection
from sklearn import metrics
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# from kerastuner.tuners import RandomSearch


# In[3]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df


# In[4]:


df.dropna(inplace=True)


# In[5]:


def fun(df):
    print(df.shape)
    print('********'*5)
    print(df.info())
    print('********'*5)
    print(df.describe())
    print('********'*5)
    print(df.isnull().sum())
fun(df)


# # Categorical Data Analysis.

# In[6]:


gender_count=df['gender'].value_counts()
gender_count


# In[7]:


plt.figure(figsize=(12,8))
gender_count.plot(kind='bar',grid=True,legend=True)
plt.show()


# In[8]:


x=pd.crosstab(df['gender'],df['ever_married'])
x


# In[9]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['gender'],hue=df['ever_married'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[10]:


x1=pd.crosstab(df['gender'],df['heart_disease'])
x1


# In[11]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['gender'],hue=df['heart_disease'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[12]:


x2=pd.crosstab(df['gender'],df['hypertension'])
x2


# In[13]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['gender'],hue=df['hypertension'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[14]:


x3=pd.crosstab(df['gender'],df['smoking_status'])
x3


# In[15]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['gender'],hue=df['smoking_status'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[16]:


x3=pd.crosstab(df['gender'],df['stroke'])
x3


# In[17]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['gender'],hue=df['stroke'],data=df)
for p in ax.patches:
  ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

#Here obviously count wise we can conclude anything so lets see in terms of proportion.
#male  = 108/2007*100 = 5.3811659192
#female = 141/2853*100 = 4.94216614090
#So, we can conclude one thing is males are more prone towards stroke.
# In[18]:


x4=pd.crosstab(df['hypertension'],df['stroke'])
x4


# In[19]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['hypertension'],hue=df['stroke'],data=df)
for p in ax.patches:
  ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

#Hypertension:
#0  = 183/4429*100 = 4.13185820
#1 = 66/432*100 = 15.27777
#So, we can conclude that if person has any external tension then he is mosty prone to stroke.
# In[20]:


x5=pd.crosstab(df['heart_disease'],df['stroke'])
x5


# In[21]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['heart_disease'],hue=df['stroke'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

#Heart Disease:
#0  = 202/4632*100 = 4.3609671
#1 = 47/229*100 = 20.524
#So, we can conclude that if person has heart disease then he is mosty prone to stroke.
# In[22]:


x6=pd.crosstab(df['ever_married'],df['stroke'])
x6


# In[23]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['ever_married'],hue=df['stroke'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

#Ever Married:
#Yes  = 220/3133*100 = 7.022023619
#No = 29/1728*100 = 1.678240
#It shows that married people are more prone to stroke.
# In[24]:


x7=pd.crosstab(df['work_type'],df['stroke'])
x7


# In[25]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['work_type'],hue=df['stroke'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[26]:


x8=pd.crosstab(df['Residence_type'],df['stroke'])
x8


# In[27]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['Residence_type'],hue=df['stroke'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()


# In[28]:


x9=pd.crosstab(df['smoking_status'],df['stroke'])
x9


# In[29]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x= df['smoking_status'],hue=df['stroke'],data=df)
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.show()

#Smoking Status:
#Formerly smoked  = 70/815*100 = 8.588
#never smoked = 90/1802*100 = 4.994
#Smokes = 42/747*100 = 5.622
#Unknown = 47/1497*100 = 3.139
#It shows that formerly smoked and current smokers are prone to strone with compare to the rest.
# # Numerical Data Analysis.

# In[30]:


sns.pairplot(df,diag_kind='kde')
plt.show()


# In[31]:


r = ["bmi","avg_glucose_level","age"]
for i in r:
    plt.figure(figsize=(12,8))
    plt.hist(x=i,bins=30,data=df,alpha=0.7, rwidth=0.85)
    plt.title(i)
    plt.show()

#Body Mass Index (BMI) is a person's weight in kilograms divided by the square of height in meters. A high BMI can be an indicator of high body fatness. BMI can be used to screen for weight categories that may lead to health problems but it is not diagnostic of the body fatness or health of an individual.
# In[32]:


x=list(df['age'].values)
hist_data=[x]
group_labels = ['age_distribution']
figure=ff.create_distplot(hist_data,group_labels)
figure.show()


# In[33]:


x11=list(df['avg_glucose_level'].values)
hist_data=[x11]
group_labels = ['avg_glucose_level_distribution']
figure=ff.create_distplot(hist_data,group_labels,bin_size=10)
figure.show()


# In[34]:


x12=list(df['bmi'].values)
hist_data=[x12]
group_labels = ['bmi_distribution']
figure=ff.create_distplot(hist_data,group_labels,bin_size=2)
figure.show()


# In[35]:


px.box(data_frame = df, x = "age", width = 800,height = 300)


# In[36]:


px.box(data_frame = df, x = "avg_glucose_level", width = 800,height = 300)


# In[37]:


px.box(data_frame = df, x = "bmi", width = 800,height = 300)


# In[38]:


plt.figure(figsize=(12,8))
g1= sns.kdeplot(df['age'][(df["stroke"] == 0)] , color="Red", shade = True)
g2 = sns.kdeplot(df['age'][(df["stroke"] == 1)], ax =g1, color="Blue", shade= True)
g1.set_xlabel('Age')
g1.set_ylabel("Frequency")
plt.show()

#Mostly old age people are prone to get stroke with copare to young peoples.
# In[39]:


plt.figure(figsize=(12,8))
g1= sns.kdeplot(df['avg_glucose_level'][(df["stroke"] == 0)] , color="Red", shade = True)
g2 = sns.kdeplot(df['avg_glucose_level'][(df["stroke"] == 1)], ax =g1, color="Blue", shade= True)
g1.set_xlabel('avg_glucose_level')
g1.set_ylabel("Frequency")
plt.show()

#Here we can see people who are getting stroke has avg glucose level of 90-250.
# In[40]:


df.columns


# In[41]:


plt.figure(figsize=(12,8))
g1= sns.kdeplot(df['age'][(df["hypertension"] == 0)] , color="Red", shade = True)
g2 = sns.kdeplot(df['age'][(df["hypertension"] == 1)], ax =g1, color="Blue", shade= True)
g1.set_xlabel('age')
g1.set_ylabel("Frequency")
plt.show()


# In[42]:


plt.figure(figsize=(12,8))
g1= sns.kdeplot(df['age'][(df["heart_disease"] == 0)] , color="Red", shade = True)
g2 = sns.kdeplot(df['age'][(df["heart_disease"] == 1)], ax =g1, color="Blue", shade= True)
g1.set_xlabel('age')
g1.set_ylabel("Frequency")
plt.show()


# In[43]:


plt.figure(figsize=(12,8))
sns.scatterplot(x=df['age'],y=df['bmi'],hue=df['stroke'])
plt.show()


# In[44]:


plt.figure(figsize=(12,8))
sns.scatterplot(x=df['age'],y=df['avg_glucose_level'],hue=df['stroke'])
plt.show()


# In[45]:


plt.figure(figsize=(12,8))
sns.scatterplot(x=df['bmi'],y=df['avg_glucose_level'],hue=df['stroke'])
plt.show()


# In[46]:


data_male=df[df['gender']=='Male']
print('Median BMI of male with age less than 30 : ',data_male[data_male['age']<30]['bmi'].median())
print('Median BMI of male with age more than 30 and less than 50 : ',data_male[(data_male['age']>30) & (data_male['age']<50)]['bmi'].median())
print('Median BMI of male with age greater than 50 : ',data_male[data_male['age']>50]['bmi'].median())


# In[47]:


data_female=df[df["gender"]=="Female"]
print('Median BMI of Female with age less than 30 : ',data_female[data_female['age']<30]['bmi'].median())
print('Median BMI of Female with age more than 30 and less than 50 : ',data_female[(data_female['age']>30) & (data_female['age']<50)]['bmi'].median())
print('Median BMI of Female with age greater than 50 : ',data_female[data_female['age']>50]['bmi'].median())


# # Feature Engineering

# In[48]:


df.corr()


# In[49]:


df.columns


# In[50]:


df.drop(df[df['gender']=='Other'].index,inplace=True)


# In[51]:


df.drop('id',axis=1,inplace=True)


# In[52]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['work_type'] = le.fit_transform(df['work_type'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])


# In[53]:


df.drop(['smoking_status','ever_married','work_type','Residence_type'],axis=1,inplace=True)


# In[54]:


X = df.drop(['stroke'], axis=1)
y = df['stroke']


# In[55]:


from imblearn.over_sampling import SMOTE
from collections import Counter

smt = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
X, y = smt.fit_resample(X, y)

counter = Counter(y)
print('After',counter)


# In[56]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=0)


# In[73]:


m1=LogisticRegression()
m2=RandomForestClassifier(random_state=0)
m3=ExtraTreesClassifier(random_state=0)


# In[74]:


m2=RandomForestClassifier(random_state=0)
parameter={'n_estimators': np.arange(1,50),'criterion': ['entropy','gini']}
GS=GridSearchCV(m2,parameter,cv = 3,scoring = 'recall')
GS.fit(X,y)


# In[75]:


GS.best_params_


# In[77]:


m3=ExtraTreesClassifier(random_state=0)
parameter={'n_estimators': np.arange(1,50),'criterion': ['entropy','gini']}
GS1=GridSearchCV(m3,parameter,cv = 3,scoring = 'recall')
GS1.fit(X,y)


# In[78]:


GS1.best_params_


# In[79]:

m1=LogisticRegression()
m2=RandomForestClassifier(n_estimators=35,criterion='entropy',random_state=0)
m3=ExtraTreesClassifier(n_estimators=19,criterion='gini',random_state=0)


# In[80]:


kf=KFold(n_splits=3,shuffle=True,random_state=0)
for model,name in zip([m1,m2,m3],['LR','RF','ETC']):
    roc_auc=[]
    for train,test in kf.split(X,y):
        X_train,X_test=X.iloc[train],X.iloc[test]
        
        y_train,y_test=y.iloc[train],y.iloc[test]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        fpr,tpr,_=roc_curve(y_test,y_pred)
        roc_auc.append(auc(fpr,tpr))
    print('AUC scores : %.02f (+/- %.05f) [%s]'% (np.mean(roc_auc),
                                                 np.var(roc_auc,ddof=1),name))


# In[86]:


sns.distplot(y_test-y_pred)


# In[81]:


pd.DataFrame(m2.feature_importances_,index=X.columns).sort_values(ascending=False,by=0)


# In[83]:


plt.figure(figsize=(12,8))
accuracy=accuracy_score(y_test, y_pred)*100
print("Accuracy Score: ","{0:.2f}".format(accuracy))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)),annot=True,fmt="g", cmap='viridis')
print(metrics.confusion_matrix(y_test, y_pred))
plt.show()


# In[84]:


print("Random Forest Classifier report \n", classification_report(y_test, y_pred))


# In[85]:


pickle.dump(m2,open('model.pkl','wb'))


# In[ ]:






