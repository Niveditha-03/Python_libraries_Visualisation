#!/usr/bin/env python
# coding: utf-8

# In[351]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, binom, geom


# In[122]:


df = pd.read_csv("BC_1-Aerofit_Treadmill.csv")
df


# In[123]:


df.shape


# In[125]:


df.describe()


# In[127]:


df.describe(include = 'object')


# In[117]:


df.nunique()


# In[118]:


df.info()


# In[119]:


df.groupby("Product").ngroups

#1. Initial Observation from Data Basic Metrics,
OBS-1:
1. There are 180 entires(180 rows, 9 colmns) in the data having people buying 3 groups of treadmill Products(Kp281, KP481, KP781).

2. Its a complete data set with no nulls.

3. 3 products bought by Male/Female with Marital status of single/partnered.

4. Age of min 18 to max of 50 has bought th treadmills with mean = 28 and 75th percentile==33(indicating 75 percent of people who bought treadmill are 33 and below)

5. Education (min 15 -21 years of education) with more number of users having 15 years of education with 1 std showing the spread of data to be less that it forms steeper graph compares to others 

6. DTYPE : Prodict, Marital status and Gender are object data types which forms categorical data while others are numerical(integer) data
# In[130]:


df["Product"].value_counts()


# In[150]:


df["Product"].unique()


# In[151]:


#Below shows head counts on each AGE 
df["Age"].value_counts()


# In[154]:


df["Gender"].value_counts()


# In[156]:


df["MaritalStatus"].value_counts().

#2. Observations:
1. There are 3 unique products ['KP281', 'KP481', 'KP781'] with more number of people (80) using KP281, 60 using KP481 and 40 using KP781.

2. Sales of KP281 is higher since the cost of the product is lower than the others. 

3. Also On an average is 3 times a week.

4. We can observe that Paqtnered people has bought more and also when comparing the gender, men has bought the treadmills in a higher ratio than women
# In[166]:


#3. Data visualisation for more understanding
##UNIVARIATE ANALYSIS -displot (N)

sns.distplot


# In[179]:


sns.displot(df["Age"], kde = True, color = "Grey", )


# In[61]:


##UNIVARIATE ANALYSIS -displot (Age - Numerical)
sns.boxplot(x = df["Age"])
plt.title("Fig-1:Age")


# In[180]:


sns.histplot


# In[66]:


#1. ##UNIVARIATE ANALYSIS -histplot (Age - Numerical)
sns.histplot( x = df["Age"])
plt.title("Histogram chart of Customer's Age")


# In[62]:


##BIVARIATE ANALYSIS -countplot (Gender vs Age - (Categorical-Numerical)

countplot to visualise the men vs women usage of tradmills
sns.countplot( x = df["Gender"], hue = df["Usage"])
plt.title("Fig-2:Gender vs Usage")


# In[181]:


##BIVARIATE ANALYSIS -countplot (Marital_status vs Product - (Categorical-Categorical)

sns.countplot(x = df["MaritalStatus"], hue = df["Product"])
plt.title("Fig-3:Count of Marital_status vs Product ")

#3. Visual Analysis:
Insights From Figure-1 and Figure-2 and figure-3: 
1. From boxplot,people of age from 19 - 46 uses Aerofit trademill and >47 are outliers. But looks like most of the people are in range of 23 to 34. The histograms clearly indicates the fact that people 25 age people are in more number(arnd 50) and <10 people of age 50 uses treadmill.

2. The median lies around 26 which indicates people ofage arnd 26 buys the treadmill a lot than than others

3. Usage of treadmill by men is higher compared to women

4. From figure-4, it shows, that marital status has effect on tradmil product sales wherein Patnered people buys more than single which could be more due to health awareness and economical status.
# In[185]:


##pairplot
sns.pairplot(df)


# In[226]:


df


# In[276]:


#4. Outliers using IQR 
data_mean, data_std = np.mean(df["Age"]), np.std(df["Age"])
print("data_mean:", data_mean, "data_std:", data_std)

cut_off = data_std * 1.3
lower, upper = data_mean - cut_off, data_mean + cut_off

print("lower_cutoff:", lower, "upper_cutoff:", upper)

outliers = [x for x in df["Age"] if x < lower or x > upper ]
print("The num of outliers are:",  len(outliers), "Outliers in age ar:", outliers)

There are 30 outliers and the Inter Quartile Range is between 19.8 to 37.8
# In[208]:


pd.crosstab


# In[249]:


percentage_maritalstatus_product = pd.crosstab(df["Product"], df["MaritalStatus"],  margins = True, normalize = True) *100
percentage_maritalstatus_product

1. 44% of people bought KP281, 33% people bought KP481, 22% bought KP781. Out of which partnered people has upper hand over unmarried persons. 
# In[233]:


type(percentage_maritalstatus_product)


# In[256]:


#people who bought 70%lesser in kp281
df.describe()


# In[283]:


#Relatinship between income and product
sns.boxplot( x = df["Income"], y = df["Product"])

1. Both the variables are well related. since the cost of order of products are KP781 > KP481 >KP281. People having higher income prefers to buy the product KP781.
2. People with income 55K and lower prefer KP481 and KP281 but people with mid range of salaries buys 481. So it shows the income is directly related with product sales.
# In[302]:


sns.scatterplot


# In[303]:


sns.scatterplot(x = df["Product"], y = df["Income"], hue = df["Age"])


# In[318]:


sns.catplot


# In[328]:


#Comparison of usage and Fitness with Product
sns.boxplot(data = df, x = df["Product"], y = df["Usage"], hue = df["Fitness"] )

KP781 has advanced features, hence people who bought the KP781 and used 4-5 times a week are in excellent shape of fitness.
# In[329]:


df


# In[369]:


ser = pd.DataFrame( data = [["KP281", 1500], ["KP481",  1750], ["KP781", 2500]], columns = ["Product", "Cost"])
ser

Recommendations and Insights:
1. The features and cost are higher in KP781 > KP481 > KP281. People with higher income(especially partnered )has higher chance of buying KP781. So when married people come to the store, we can explain the advance feature and show them statistical results of fitness of KP781 where fitness rate is higher in this product and try to sale. 
2. Number of male who bought the tradmills are higher than female which could be more of health conciousness. So If the customer is around the age of 20-28, help them buy the correct tradmill.
3. Usage and fitness is directly proportional, so better the usage better will be the fitness.
4. People who uses their aerofit treadmill on regular basis are considered to be healthy in shape.
# In[388]:


sns.heatmap


# In[391]:


sns.heatmap(df.corr())


# In[ ]:




