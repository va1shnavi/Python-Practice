#!/usr/bin/env python
# coding: utf-8

# ### **Banking Analytics | Problem Statement: Increase the effectiveness of the bank's telemarketing campaign**
# 
# **Name: Vaishnavi Panchal**
# 
# **Dataset Source: kaggle**
# 
# **Email: vaishpanchal12@gmail.com**

# ##### **Import Libraries**

# In[259]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ##### **Import the dataset**

# In[260]:


dataset = pd.read_csv("bank-full.csv")


# In[261]:


dataset.head()


# In[262]:


# dataset["job"].str.contains('unknown').sum()  
# checks in that column how many unknown termed entries are there


# In[263]:


print("Number of null values:\n", dataset.isnull().sum())
print("Number of duplications: ", dataset.duplicated().sum())


# ##### **We check if the data is properly balanced**

# In[264]:


dataset['y'].hist()  # The data is imbalanced


# In[265]:


fig = plt.figure(figsize = (15, 15))
ax = fig.gca()  # get current axis
dataset.hist(ax = ax)
plt.show()


# ## **The data isn't normalised, balanced and has categorial varibles to be taken care of**

# ##### **First we transform the columns with binary values, "yes" maps to 1 and "no" maps to 0**

# In[266]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.iloc[:, 4] = le.fit_transform(dataset.iloc[:, 4])  # default
dataset.iloc[:, 6] = le.fit_transform(dataset.iloc[:, 6])  # housing
dataset.iloc[:, 7] = le.fit_transform(dataset.iloc[:, 7])  # loan
dataset.iloc[:,-1] = le.fit_transform(dataset.iloc[:,-1])  # dependent variable

# dt['default'] = dt['default'].map({'yes': 1, 'no': 0})   manually doing it


# #### **The columns that don't contribute to the analysis are removed**
# 
# *The columns contact, day, month in the dataset seem to be useless, so they are dropped from the dataset*

# In[267]:


dataset.drop(["contact",	"day",	"month"	],1, inplace = True) # remove columns contact, day, month as they arent useful


# ##### **The categorical columns are transformed and dummy variables are created**

# In[268]:


dummy = pd.get_dummies(dataset[["job", "marital", "education", "poutcome"]], drop_first=True)
dataset = pd.concat([dataset, dummy],axis=1)


# In[269]:


dataset.drop(["job","marital", "education", "poutcome"], 1, inplace=True)


# ##### **Now the dataset has only numerical values**

# In[270]:


dataset.head()


# In[271]:


dataset.info()


# ##### **Handling the outliers**
# 

# In[272]:


# View the outliers using boxplot

plt.figure(figsize=(10,10))
plt.subplot(6, 1, 1)
sns.boxplot(dataset["age"])
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(6, 1, 2)
sns.boxplot(dataset["balance"])
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(6, 1, 3)
sns.boxplot(dataset["duration"])
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(6, 1, 4)
sns.boxplot(dataset["campaign"])
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(6, 1, 5)
sns.boxplot(dataset["pdays"])
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(6, 1, 6)
sns.boxplot(dataset["previous"])
plt.show()


# In[273]:


dataset.shape  # the dimensions before the outliers are taken care of


# In[274]:


# Outlier handling 

from scipy import stats
z = np.abs(stats.zscore(dataset[['age','balance','duration','campaign','pdays','previous']]))
dataset = dataset[(z < 3).all(axis=1)]  # z < 3 are all the rows which aren't outliers (z> 3 will be the outliers rows)


# In[275]:


dataset.shape   # the dimensions after handling the outliers


# In[276]:


# View the outliers using boxplot

plt.figure(figsize=(10,7))
plt.subplot(6, 1, 1)
sns.boxplot(dataset["age"])
plt.show()

plt.figure(figsize=(10,7))
plt.subplot(6, 1, 2)
sns.boxplot(dataset["balance"])
plt.show()

plt.figure(figsize=(10,7))
plt.subplot(6, 1, 3)
sns.boxplot(dataset["duration"])
plt.show()

plt.figure(figsize=(10,7))
plt.subplot(6, 1, 4)
sns.boxplot(dataset["campaign"])
plt.show()

plt.figure(figsize=(10,7))
plt.subplot(6, 1, 5)
sns.boxplot(dataset["pdays"])
plt.show()

plt.figure(figsize=(10,7))
plt.subplot(6, 1, 6)
sns.boxplot(dataset["previous"])
plt.show()


# In[277]:


dataset.head()


# In[278]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(dataset.corr(),vmin=-1, vmax=1,annot=True,cmap="YlGnBu")
ax.set_title('Multi-Collinearity of the Attributes')
plt.show()


# ##### **We remove the irrelvent variables by checking their dependencies**

# In[279]:


x = dataset.drop(["y"], axis = 1)
y = dataset["y"]


# In[280]:


import statsmodels.api as sm

lor = sm.Logit(y,x).fit()
lor.summary()


# In[281]:


x.drop(["education_unknown", "poutcome_other"], axis=1, inplace=True)


# In[282]:


lor = sm.Logit(y,x).fit()
lor.summary()


# In[283]:


x.drop(["previous", "default"], axis=1, inplace=True)


# In[284]:


lor = sm.Logit(y,x).fit()
lor.summary()


# In[285]:


x.drop(["education_secondary"], axis=1, inplace=True)


# In[286]:


lor = sm.Logit(y,x).fit()
lor.summary()


# ##### **To balance the data, we use the undersampling technique- Near Miss**

# In[287]:


# Undersample imbalanced dataset with NearMiss-1
from collections import Counter
from imblearn.under_sampling import NearMiss


u = NearMiss()
# summarize class distribution
counter_before = Counter(y)
print(counter_before)
# define the undersampling method

undersample = NearMiss(version=1, n_neighbors=5)
# transform the dataset
x, y = undersample.fit_resample(x, y)
# summarize the new class distribution
counter_after = Counter(y)
print(counter_after)


# In[288]:


labels = [0, 1]
plt.figure(figsize=(10,6))
plt.subplot(1, 2, 1)
sns.barplot(labels, list(counter_before.values()))
plt.title("Numbers before balancing the dataset")
plt.subplot(1, 2, 2)
sns.barplot(labels, list(counter_after.values()))
plt.title("Numbers after balancing the dataset")
plt.show()


# In[289]:


fig = plt.figure(figsize = (15, 15))
ax = fig.gca()  # get current axis
pd.DataFrame(x).hist(ax = ax)
plt.show()


# **The data is cleaned and now ready to actually work with**
# 
# ## **We will try to fit the dataset with three classification process- Logistic Regression, RandomForest Classification and the Naive Bayes Classification. The one which gives gives the best results will be preferred for the given dataset**

# # **1. Logistic Regression**

# ##### **Splitting the dataset**

# In[290]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# ##### **Scaling the dataset**

# In[291]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ##### **Fitting the model**

# In[292]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


# ##### **Predicting the test set**

# In[293]:


y_pred = classifier.predict(x_test)


# ### **Metric results**

# In[294]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))


# # **2. Random Forest Classification**

# ##### **Splitting the dataset**

# In[295]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# ##### **Scaling the dataset**

# In[296]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ##### **Fitting the model**

# In[297]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# ##### **Predicting the test set**

# In[298]:


y_pred = classifier.predict(x_test)


# ### **Metric results**

# In[299]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))


# # **3. Naive Bayes Classification**

# ##### **Splitting the dataset**

# In[300]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# ##### **Scaling the dataset**

# In[301]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ##### **Fitting the model**

# In[302]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# ##### **Predicting the test set**

# In[303]:


y_pred = classifier.predict(x_test)


# ### **Metric results**

# In[304]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))


# ### **After trying three different classification approaches we realise that the Random Forest Classification gives better accuracy, precision and recall; compared to Logistic Regrssion and Naive Bayes Classification.**
# 
# **Hence, for the given dataset the best approach will be Random Forest Classification.**
