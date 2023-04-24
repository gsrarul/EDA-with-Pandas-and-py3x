#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a=[1,2,2,2,2,2,3,4,5,6,7,8,10,11,12,13,14,15]


# In[3]:


def get_mean (my_data):
    return sum(my_data)/(len(my_data)*1.0)


# In[4]:


get_mean(a)


# In[5]:


my_np_list = np.asarray(a)


# In[6]:


np.mean(my_np_list)


# In[7]:


b=np.asarray(a)


# In[8]:


np.mean(b)


# In[9]:


c= np.mean(b)


# In[10]:


c


# In[11]:


a.append(50)


# In[12]:


np.mean(a)


# In[13]:


np.median(a)


# In[14]:


a


# In[15]:


from scipy.stats import mode


# In[16]:


mode(a)


# In[17]:


max(a)


# min(a)

# In[18]:


min(a)


# In[19]:


a


# In[20]:


np.var(a)


# In[21]:


np.std(a)


# In[22]:


import numpy as np


# In[23]:


import seaborn as sns


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


#First argument is mean
#Second argument is Standard Deviation
#Third argument is Size 
x = np.random.normal (50,5,3000)


# In[26]:


x


# In[27]:


print("Max is {}".format(max(x)))
print("Min is {}".format(min(x)))
print("Max - Min is {}".format(max(x)-min(x)))


# # Seaborn Library 
# 
# sns.distplot (x, kde=False, bins=5,norm_hist=True, hist_kms=dict(edgecolor="k",linewidth=2))
# plt.xlabel('Intervals')
# plt.ylabel('Frequency')
# plt.show()

# In[28]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


x = np.random.normal (50,5,3000)


# In[30]:


print("Max is {}".format(max(x)))
print("Min is {}".format(min(x)))
print("Max - Min is {}".format(max(x)-min(x)))


# # Histogram
# sns.distplot (x, kde=False,bins=10 , hist_kws = dict(edgecolor="k",linewidth=2))
# plt.xlabel('Intervals')
# plt.ylabel('Frequency')
# plt.show()

# In[31]:


#Relative Frequency Histogram Plot
sns.distplot (x, kde=False,
              bins=30,
              norm_hist=True, 
              hist_kws = dict(edgecolor="k",linewidth=2))
plt.xlabel('Intervals')
plt.ylabel('Relative Frequency')
plt.show()


# In[32]:


#Density Plot
sns.distplot(x,kde=True,bins=30,
             norm_hist=True,
             hist_kws=dict(edgecolor='r',linewidth=2))
plt.xticks(range(28,70,4))
plt.ylabel('Density')
plt.show()


# In[33]:


#Density plot curve mean. mode ,medium
plt.axvline(x.mean(),color='b',linestyle='dashed',linewidth=1)
plt.axvline(np.median(x),color='r',linestyle='dashed',linewidth=4)
sns.distplot(x,kde=True,bins=30,hist=False)
plt.ylabel('Density')
plt.show()


# In[34]:


np.median(x)


# In[35]:


np.mean(x)


# In[36]:


a = [1,2,2,2,4,5,6,6,6,9,10,10,10,11,14,14,15,18,19,20]


# In[37]:


import numpy as np


# In[38]:


p25=np.percentile(a,25)
p25


# In[39]:


p90=np.percentile(a,90)
p90


# # Quartile and percentile in Box Plot using seaborn

# In[40]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

my_list=[1,2,2,2,4,5,6,6,6,9,10,10,10,11,14,14,15,18,19,20]

#Convert the list into data frame
df = pd.DataFrame(my_list,columns=['Sample data'])

#plot using seaborn
sns.boxplot(data=df)

#add ylabel

plt.ylabel('Data points')


# In[41]:


import numpy as np
np.percentile(my_list,[25,50,75])


# In[ ]:





# # finding missing values

# In[42]:


import pandas as pd
import numpy as np


# In[43]:


dummy_df = pd.read_csv("C:\\Users\\arulkumar\\Desktop\\Dummy_data.csv")


# In[44]:


dummy_df


# # df.describe()
# 
# dummy_df.describe()

# In[45]:


dummy_df.describe()


# In[46]:


len(dummy_df)


# # Since NaN is considered float , the data is converted to float type

# # boolean map **   .isna() and .notna()

# In[47]:


dummy_df.isna()


# In[48]:


dummy_df.notna()


# In[49]:


dummy_df.isnull()


# In[50]:


dummy_df.notnull()


# # Finding the count of missing value in each column 

# In[51]:


dummy_df.isnull().sum()


# In[52]:


import pandas as pd


# In[53]:


string_dummy_df = pd.read_csv('C:\\Users\\arulkumar\\Desktop\\dummy_str_data.csv')


# In[54]:


string_dummy_df

#Sometimes information for a column is insufficient to assign a value
# # Missing data for datatime

# In[55]:


time_df=pd.read_csv('C:\\Users\\arulkumar\\Desktop\\Birthday.csv')


# In[56]:


time_df


# In[57]:


type(time_df['Birthday'][0])


# In[58]:


time_df['Birthday']=pd.to_datetime(time_df['Birthday'])


# In[59]:


time_df


# # NaN values are considered to be float by default in pandas
# # in datetime format , missing value denoted by NaT 
# 

# In[60]:


time_df.isnull().sum()


# In[ ]:





# # Handling missing values

# In[61]:


import pandas as pd
import numpy as np


# # #1 Ignoring Missing data

# In[62]:


dummy_data=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\Dummy_data.csv")


# In[63]:


dummy_data


# # 1.1 Ignoring rows with missing columns

# In[64]:


dummy_data


# In[65]:


removed_na_df=dummy_data.dropna()


# In[66]:


removed_na_df


# # #thresh=x attribute tells df.dropna() keep rows with atleast 'x' Non - null values

# In[67]:


birthday_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\Birthday.csv")


# In[68]:


birthday_df


# In[69]:


birthday_df.dropna(thresh=4)


# In[70]:


birthday_df.dropna(thresh=6)


# # #Dropping columns by percentage of missing values

# #drop columns with say over 40% empty values

# In[71]:


birthday_df.dropna(axis=1,thresh=int(0.4*len(birthday_df)))

##Drop columns with 60% empty values
# # #Axis = 0 represent row and 1 is column

# In[72]:


birthday_df.dropna(axis=1,thresh=int(0.6*len(birthday_df)))


# # # Imputing values
# 

# In[73]:


#Filling with Generic values


# dummy_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\Dummy_data.csv",index_col=0)

# In[74]:


dummy_df


# In[75]:


dummy_df.fillna(-1)


# In[76]:


##dummy_data.fillna(inplace=True)
##****Note*** Changes will be permanently reflected only when inplace is True 


# # Backward fill 

# In[77]:


#Use the next non-null value to fill 


# In[78]:


dummy_df.bfill()


# # Forward fill

# ## use the previous non-null value to fill

# In[79]:


dummy_df.ffill()


# # # Filling with Central tendency

# In[80]:


dummy_df


# In[81]:


mean_age=dummy_df['Age'].mean()
mean_height=dummy_df['Height(cm)'].mean()


# In[82]:


map_dict= {'Age':mean_age,'Height(cm)':mean_height}
map_dict


# In[83]:


dummy_df.fillna(value=map_dict)


# # # Imputing values based on condition

# In[84]:


weight_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\Dummy_data.csv")


# In[85]:


weight_df


# In[86]:


weight_df.groupby("Sex")


# In[87]:


for x in weight_df.groupby("Sex"):
    print("Printing Group")
    print(x)
    print("\n\n")


# In[88]:


weight_df["Weight(kg)"]= weight_df.groupby("Sex").transform(lambda x: x.fillna(x.mean()))


# # #Missing values in Titanic data set

# In[89]:


import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
get_ipython().run_line_magic('matplotlib', 'inline')
clear_output()


# In[90]:


titanic_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\titanic-data.csv")


# In[91]:


titanic_df


# In[92]:


titanic_df.head()


# In[93]:


titanic_df.tail()


# In[94]:


titanic_df.columns


# 1.Survived - 0 is No and 1 is Yes
# 2.Pclass 1-upper class 2 - middle class and 3 - lower class
# 3.SibSp - No of siblings and spouses of the passenger aboard
# 4.Parch - No of parents and spouses of the passenger aboard
# 5.Embarked - Port of embarkation of the passenger ( c = Cherbourg , Q= Queenstown , S = Southamton

# In[95]:


titanic_df.describe()


# In[96]:


titanic_df.info()


# ##Three  columns have missing values : Age , Cabin and Embarked

# 1st Cabin column 

# In[97]:


cabin_df=titanic_df['Cabin']


# In[98]:


percentage_missing_cabin=(cabin_df.isnull().sum()/(len(titanic_df)*1.0))*100


# In[99]:


percentage_missing_cabin


# # Over 77% values in this column are missing. it's better to drop this column altogether

# In[100]:


titanic_df.drop(columns=['Cabin'],inplace=True)


# In[101]:


titanic_df.columns


# In[102]:


## Embarked Columns


# In[103]:


embarked_df=titanic_df['Embarked']


# In[104]:


embarked_df.unique()


# In[105]:


embarked_df.value_counts()


# In[106]:


644/891


# In[107]:


titanic_df['Embarked'].fillna('S',inplace=True)


# # Age Columns

# In[108]:


titanic_df['Age'].hist(bins=25)


# In[109]:


age_df=titanic_df[['Age','Sex']]


# In[110]:


age_df['Age']=age_df.groupby('Sex').transform(lambda x : x.fillna(x.mean()))


# In[111]:


age_df.hist(bins=25)


# In[112]:


age_df=titanic_df[['Age','Sex']]


# In[113]:


age_df['Age'] = age_df.groupby('Sex').transform(lambda x : x.fillna(x.median()))


# In[114]:


age_df.hist(bins=25)


# # Both of the above simply add unwanted patterns in the dataset , hence its not good idea to go ahead with them
##  We Can use machine learning based on algorithm imputing missing value like kNN,Etcs

# In[ ]:





# # Outliers

# In[115]:


import numpy as np
import scipy.stats
import pandas as pd


# In[116]:


dummy_age=[20,21,24,24,28,26,19,22,26,24,21,19,22,28,29,6,100,25,25,28,31]
dummy_height=[150,151,155,153,280,160,158,157,158,145,150,155,155,151,152,153,160,152,157,157,160,153]
dummy_df=pd.DataFrame(list(zip(dummy_age,dummy_height)),columns=['Age','Height(cm)'])


# In[117]:


dummy_df


# # Calculating Z Score using scipy.stats.zscore

# In[118]:


scipy.stats.zscore(dummy_df)


# In[119]:


scipy.stats.zscore(dummy_df['Height(cm)'])


# In[120]:


## We can use absolute values while calculating z_score


# In[121]:


z_score_height = np.abs(scipy.stats.zscore(dummy_df['Height(cm)']))
dummy_df.iloc[np.where(z_score_height>3)]


# In[122]:


z_score_age= np.abs(scipy.stats.zscore(dummy_df['Age']))
dummy_df.iloc[np.where(z_score_age>3)]

##in age column age 6 is strange value but its not refects here so we will go for mod z score 


# In[123]:


## Limitation of Z-Score 

#they assume a normal distribution curves

## also mean and standard used in the calculation can be easily distored by outliers


# In[124]:


## Modification of Z-Score 

##instead of mean use median

## Instead of standard deviation , use median absolute deviation (MAD):

## MAD = Median of (| Xi - Median ) for all Xi

## MAD is less affected by outliers , since its calculation depends on the median

## Modified Z-Score for Xi = 0.6745 * (Xi-Median)/Median Absolute 


# In[125]:


def modified_z_score(my_data):
    #first calculate median
    median_my_data = np.median(my_data)
    
    #median absolute deviation
    #median of |Xi - median of Xi for all X_i
    mad=np.median(my_data.map(lambda x : np.abs(x- median_my_data)))
    
    #Modified Z-Score
    #0.6745 * ( X i - Median Absolute Deviation)
    modified_z_score=list(my_data.map(lambda x: 0.6745 *(x - median_my_data)/mad))
    return modified_z_score


# In[126]:


modified_z_score(dummy_df['Age'])


# In[127]:


modified_z_score(dummy_df['Height(cm)'])


# In[128]:


import numpy as np


# In[129]:


mod_z_score_height=modified_z_score(dummy_df['Age'])
dummy_df.iloc[np.where(np.abs(mod_z_score_height)>=3)]


# # Calculating IQR to identify the outlier

# In[130]:


import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt


# In[131]:


dummy_age= [20,21,24,24,28,26,19,22,26,24,21,19,22,28,29,6,100,25,25,28,31]
dummy_height=[150,151,155,153,280,160,158,157,158,145,150,155,155,151,152,153,160,152,157,157,160,153]
dummy_df=pd.DataFrame(list(zip(dummy_age,dummy_height)),columns=['Age','Height(cm)'])


# In[132]:


dummy_df


# In[133]:


def get_lower_upper_bound(my_data):
    
    #Get First and third Quartile 
    q1=np.percentile(my_data,25)
    q3=np.percentile(my_data,75)
    
    #Calculate the IQR 
    iqr=q3-q1
    
    #Compute the lower and upper bound
    
    lower_bound=q1-(iqr*1.5)
    upper_bound=q3+(iqr*1.5)
    return lower_bound,upper_bound


# In[134]:


def get_outlier_iqr(my_data):
    lower_bound , upper_bound = get_lower_upper_bound(my_data)
    #Filter data less than lower bound and more than upper bound
    return my_data [np.where((my_data > upper_bound) |
                           (my_data < lower_bound))]


# In[135]:


get_outlier_iqr(dummy_df['Age'].values)


# In[136]:


get_outlier_iqr(dummy_df['Height(cm)'].values)


# In[137]:


dummy_df.boxplot(column=['Age'])


# In[138]:


dummy_df.boxplot(column=['Height(cm)'])


# # Univariable Analysis

# In[139]:


import pandas as pd


# In[140]:


df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\titanic-data.csv",index_col=0)


# In[141]:


df.head()


# In[142]:


df.describe()


# In[143]:


df.columns


# # Numerical variable
# 1. Age
# 2.SibSp
# 3.Parch
# 4.Fare

# In[144]:


df['Age']


# In[145]:


# find the unique value in Age Dataset

#Age and Fare continous variable

df['Age'].unique()


# In[146]:


df['SibSp'].unique()


# In[147]:


df['Parch'].unique()


# # SibSp and Parch are discrete variables

# # Categorical variables
# 1.Name
# 2.Survives
# 3.Pclass
# 4.Embarked
# 5.Cabin,
# 6.Sex

# Oridinal Values & Nominal Values

# In[148]:


df['Sex'].unique()


# In[149]:


df['Pclass'].unique()


# In[150]:


df['Embarked'].unique()


# In[151]:


df['Cabin'].unique()


# # Univariable Analysis

# In[152]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[153]:


dummy_age=[12,22,36,42,15,89,42,65,29,6,35,81,90,51,53,53,22,31,75,5]
dummy_sex_list=["Male","Male","Female","Female","Female","Female","Male","Other","Other","Female","Female","Female","Male","Male","Male","Female","Other","Other","Female","Male"]
age_sex_df=pd.DataFrame({'Age':dummy_age,'Sex':dummy_sex_list})


# In[154]:


age_sex_df


# In[155]:


#Categorical Variables

plt.figure()
age_sex_df['Age'].hist()


# Using another libary for ploting using seaborn

# In[156]:


# Bar Charts using seaborn (Categorical Variables)

sns.set(style="darkgrid")
ax=sns.countplot(x="Sex",data=age_sex_df)


# In[157]:


maths_marks=[20,18,15,25,17,16]
science_marks=[14,16,20,15,15,12]
english_marks=[20,18,10,15,25,10]
marks_df=pd.DataFrame({'Maths':maths_marks,'Science':science_marks,'English':english_marks})


# In[158]:


marks_df


# In[159]:


marks_df.plot.bar()


# In[160]:


#DEnsity Plot 

marks_df.plot.density()


# In[161]:


marks_df.boxplot()


# # Univariable analysis over Olympic data set 
# Source https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results

# In[162]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[163]:


atheletes_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\athlete_events.csv")
regions_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\noc_regions.csv")


# In[164]:


data_df=pd.merge(atheletes_df,regions_df,on='NOC',how='left')


# In[165]:


data_df.head()


# In[166]:


data_df.columns


# In[167]:


data_df.describe()


# In[168]:


data_df.info()


# In[169]:


sns.distplot(data_df['Age'])


# In[170]:


age_df=pd.to_numeric(data_df['Age'],errors='coerce')
age_df=age_df.dropna()
age_df=age_df.astype(int)


# In[171]:


sns.countplot(age_df)


# In[172]:


data_df.loc[data_df['Medal'].isnull()]


# In[173]:


medalist_df=data_df.loc[data_df['Medal'].notnull()]


# In[174]:


medalist_df


# In[175]:


def plot_column(my_df,col,chart_type='Histogram',dtype=int,bin_size=25):
    temp_df=pd.to_numeric(my_df[col],errors='coerce')
    temp_df=temp_df.dropna()
    temp_df=temp_df.astype(dtype)
    if chart_type=='Histogram':
        ax=sns.countplot(temp_df)
    elif chart_type=='Density':
        ax=sns.distplot(temp_df)
    xmin,xmax=ax.get_xlim()
    ax.set_xticks(np.round(np.linspace(xmin,xmax,bin_size),2))
    plt.tight_layout()
    plt.locator_params(axis='y',nbins=6)
    plt.show()


# In[176]:


plot_column(medalist_df,'Age')


# Since it not symmentrical distribution we have to calculate skewness

# # Calculating Skewness

# In[177]:


from scipy.stats import skew

age_df=pd.to_numeric(medalist_df['Age'],errors='coerce')
age_df=age_df.dropna()
age_df=age_df.astype(int)
print("Skewness is {}",format(skew(age_df)))
print("Mean is {}",format(np.mean(age_df)))
print("Meadian is {}",format(np.median(age_df)))


# In[178]:


plot_column(medalist_df,'Height',bin_size=15)


# In[179]:


height_df=pd.to_numeric(medalist_df['Height'],errors='coerce')
height_df=height_df.dropna()
height_df=height_df.astype(int)
print("Skewness is {}",format(skew(height_df)))
print("Mean is {}",format(np.mean(height_df)))
print("Meadian is {}",format(np.median(height_df)))


# In[180]:


weight_df=pd.to_numeric(medalist_df['Weight'],errors='coerce')
weight_df=weight_df.dropna()
weight_df=weight_df.astype(int)
print("Skewness is {}",format(skew(weight_df)))
print("Mean is {}",format(np.mean(weight_df)))
print("Meadian is {}",format(np.median(weight_df)))


# In[181]:


sport_df=medalist_df[~medalist_df['Sport'].isnull()]
sns.countplot(medalist_df['Sport'])


# In[182]:


sum(medalist_df['Sport'].isnull())


# In[183]:


sport_count=medalist_df['Sport'].value_counts().nlargest(25).to_frame()


# In[184]:


print(sport_count)


# In[185]:


ax=sport_count.plot.bar(y='Sport')
ax.get_legend().remove()


# In[186]:


year_count_df=data_df['Year'].value_counts().to_frame()


# In[187]:


year_count_df.sort_index(inplace=True)
ax=year_count_df.plot.bar(y='Year')
ax.get_legend().remove()


# # Scatter plots and heat maps

# In[188]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[189]:


x1=np.random.normal(10,1,200)*10


# In[190]:


sns.distplot(x1)


# # Negative Correlation with x1

# In[191]:


y1= 100-x1


# In[192]:


ax=sns.scatterplot(x1,y1)
ax.set(xlabel='x1',ylabel='y1')
plt.show()


# In[193]:


from scipy.stats import pearsonr
pearsonr(x1,y1)


# in real world , you probably see some noise in your data

# In[194]:


x2=np.random.normal(10,1,200)*10
y2=x2 + np.random.normal(40,5.2,200)


# In[195]:


ax=sns.scatterplot(x2,y2)
ax.set(xlabel='x2',ylabel='y2')
plt.show()


# In[196]:


pearsonr(x2,y2)


# In[197]:


ax=sns.regplot(x2,y2)
ax.set(xlabel='x2',ylabel='y2')
plt.show()


# # Heat Map

# In[198]:


# Create some random variate 

x3=np.random.random(200)
y3=x1+x3-20

x4=np.random.normal(100,1.5,200)
y4=x1+x3-20

data_df=pd.DataFrame({'x1':x1, 'x2':x2,'x3':x3,'x4':x4,'y1':y1,'y2':y2,'y3':y3,'y4':y4})


# In[199]:


sns.pairplot(data_df)


# In[200]:


data_df.corr()


# In[201]:


#Setting size
sns.set(rc={'figure.figsize':(8,8)})

#Drafting heatmap for correlation coefficient
ax=sns.heatmap(data_df.corr(),annot=False,linewidths=1,fmt='.2f')


# In[202]:


ax=sns.heatmap(data_df.corr(),annot=True,linewidths=1,fmt='.2f')


# In[203]:


ax=sns.heatmap(data_df.corr(method='spearman'),annot=True,linewidths=1,fmt='.2f')


# In[204]:


ax=sns.heatmap(data_df.corr(method='pearson'),annot=True,linewidths=1,fmt='.2f')


# # Bivariate Analysis on titanic dataset

# DataSource:  https://www.kaggle.com/competitions/titanic/data

# In[205]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[206]:


titanic_data_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\titanic-data.csv")


# In[207]:


titanic_data_df.head()


# ![image.png](attachment:image.png)

# Study of Titanic Data Set

# In[208]:


g=sns.countplot(x='Sex',hue='Survived',data=titanic_data_df)


# In[209]:


#Categorical ploting of Emabarked and Survived

g=sns.catplot(x='Embarked',col='Survived',data=titanic_data_df,kind="count",height=3,aspect=.7)


# In[210]:


# Without Categorical ploting of Emabarked and Survived
g=sns.countplot(x='Embarked',hue='Survived',data=titanic_data_df)


# In[211]:


g=sns.catplot(x='Pclass',col='Survived',data=titanic_data_df,kind="count",
             height=3,aspect=.5)


# # Add new column -Family Column

# adding a new column "Family Size" which will be SibSP and Parch + 1 

# In[212]:


def add_family(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] +1
    return df
titanic_data_df=add_family(titanic_data_df)
titanic_data_df.head()


# In[213]:


g=sns.countplot(x="FamilySize",hue="Survived",data=titanic_data_df)


# # Add new column - age group

# In[214]:


age_df=titanic_data_df[~titanic_data_df['Age'].isnull()]

# make bin and group all passengers into these bins and store as new values 

age_bins=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79']
age_df['ageGroup']=pd.cut(titanic_data_df.Age,range(0,81,10),right=False,labels=age_bins)


# In[215]:


age_df[['Age','ageGroup']].head()


# In[216]:


sns.countplot(x='ageGroup',hue='Survived',data=age_df)


# # Data Resource - https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings

# In[217]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[218]:


data_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\Video_Games_Sales_as_at_22_Dec_2016.csv")


# In[219]:


data_df.head()


# In[220]:


data_df.info()


# In[221]:


data_df.describe()


# In[222]:


data_df.columns


# the most obivious things comes to mind is that , is any column related to global series for a games

# In[223]:


sns.pairplot(data_df)


# In[224]:


sns.set(rc={'figure.figsize':(8,8)})
sns.heatmap(data_df.corr(),annot=True,fmt='.2f')


# lets try to focus on sales columns only 

# In[225]:


sns.heatmap(data_df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].corr(),annot=True,fmt='.2f')


# lets try focusing on User and critics score  

# In[226]:


ax= sns.scatterplot(x=data_df['Critic_Score'],y=data_df['User_Score'])
ymin,ymax=ax.get_ylim()
ax.set_yticks(np.round(np.linspace(ymin,ymax,25,2)))
plt.tight_layout()
plt.locator_params(axis='y',nbins=6)
plt.show()


# In[227]:


score_df=data_df[['Critic_Score','User_Score']]
score_df=score_df[score_df['User_Score']!='tbd']
score_df['User_Score']=pd.to_numeric(score_df['User_Score'],errors='coerce')
score_df.dropna(how='any',inplace=True)


# In[228]:


score_df.info()


# In[229]:


sns.scatterplot(x=score_df['Critic_Score'],y=score_df['User_Score'])


# In[230]:


score_df.corr()


# In[231]:


score_df.corr(method='spearman')


# In[232]:


score_df.corr(method='pearson')


# Lets move on to Game

# In[233]:


genre_group=data_df.groupby('Genre').size()
genre_group.plot.bar()


# In[234]:


data_df['Rating'].uniqueque()


# In[235]:


g=sns.catplot(x='Genre',hue='Rating',
             data=data_df,kind="count",height=10);


# In[236]:


count_year_gen=pd.DataFrame({'count':data_df.groupby(['Genre','Year_of_Release']).size()}).reset_index()
print(data_df.groupby(['Genre','Year_of_Release']).size())


# 
# # Release by Genre

# In[237]:


ax=sns.boxplot(x="count",y="Genre",data=count_year_gen,whis=np.inf)


# In[238]:


sns.set(rc={'figure.figsize':(20,8)})
ax=sns.lineplot(x='Year_of_Release',y='count',hue='Genre',data=count_year_gen)


# We can't deep drive analysis thru this so we are importing other lib called Bokeh

#     Code Modified from : https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/legends.html

# In[239]:


from bokeh.palettes import Spectral11
from bokeh.plotting import figure,show
p=figure(plot_width=800,plot_height=550)
p.background_fill_color="beige"

p.title.text='Click on legend entries to hide the corresponding lines'
import random
legend_list=[]
for genre_id in count_year_gen['Genre'].unique():
    color=random.choice(Spectral11)
    df=pd.DataFrame(count_year_gen[count_year_gen['Genre']==genre_id])
    p.line(df['Year_of_Release'],df['count'],line_width=2,alpha=0.8,color=color,legend=genre_id)
p.legend.location='top_left'
p.legend.click_policy='hide'
show(p)


# In[240]:


genre_region_jp = pd.DataFrame({'Sales': data_df.groupby('Genre')['JP_Sales'].sum()}).reset_index()
sns.barplot(x='Genre', y='Sales', data=genre_region_jp)


# In[241]:


genre_region_na = pd.DataFrame({'Sales': data_df.groupby('Genre')['NA_Sales'].sum()}).reset_index()
sns.barplot(x='Genre', y='Sales', data=genre_region_na)


# In[242]:


genre_region_eu = pd.DataFrame({'Sales': data_df.groupby('Genre')['EU_Sales'].sum()}).reset_index()
sns.barplot(x='Genre', y='Sales', data=genre_region_eu)


# In[243]:


genre_region_othersales = pd.DataFrame({'Sales': data_df.groupby('Genre')['Other_Sales'].sum()}).reset_index()
sns.barplot(x='Genre', y='Sales', data=genre_region_othersales)


# In[244]:


data_df.groupby('Genre')['Publisher'].apply(lambda x:x.value_counts().index[0])


# # Multi Variate Analysis on titanics dataset

# In[245]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[246]:


titanic_data_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\titanic-data.csv")


# In[247]:


titanic_data_df.head()


# In[248]:


fig1=plt.figure()
first_axis=fig1.add.subplot(111)
second_axis=first_axis.twinx()

#the scales of the y axis on the left and right hand side are different

survived_df=titanic_data_df.groupby(['Embarked']).mean()[['Survived']]
fare_df=titanic_data_df.groupby(['Embarked']),mean()[['Fare']]

#Plot data for fare and plot of embarkment 

fare_df.plot(kind='bar',grid=True,ax=first_axis,width=0.2,position=0)
survived_df.plot(kind='bar',color="yellow",grid=True,a=second_axis,width=0.2,position=0)

#set axis 

first_axis.set_ylabel('Average_Fare_amount')
second_axis.set_ylabel('Survived%')

#display legend

first_axis.legend(["Average"])


# # # Multi Variate Analysis on Pokemondataset

# In[249]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[250]:


pokemon_df=pd.read_csv("C:\\Users\\arulkumar\\Desktop\\pokemon.csv")


# In[251]:


pokemon_df.head()


# In[252]:


pokemon_df.info()


# In[253]:


pokemon_df.describe()


# # Pokemon dataset type 2 has some missing value 

# In[254]:


pokemon_df['Type 2'].fillna(value='NA',inplace=True)


# In[255]:


pokemon_df.info()


# In[256]:


legendary_df=pokemon_df[pokemon_df['Legendary']==True]


# In[257]:


legendary_df.head()


# In[258]:


ax=sns.countplot(pokemon_df['Type 1'])
g=ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# Type 2 Pokemon countplot Since NA is High in Type 2 so we are filter out that and ploting count plot

# In[259]:


ax=sns.countplot(pokemon_df[pokemon_df['Type 2']!='NA']['Type 2'])
g=ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# In[260]:


plt.subplots(figsize=[20,5])
plt.title('Attack by Type1')
sns.boxplot(x="Type 1",y="Attack",data=pokemon_df)


# In[261]:


plt.subplots(figsize = (20,5))
plt.title('Attack by Type2')
sns.boxplot(x = "Type 2", y = "Attack",data = pokemon_df)


# In[262]:


plt.subplots(figsize = (20,5))
plt.title('Defense by Type1')
sns.boxplot(x = "Type 1", y = "Defense",data = pokemon_df)


# In[263]:


plt.subplots(figsize = (20,5))
plt.title('Defense by Type2')
sns.boxplot(x = "Type 2", y = "Defense",data = pokemon_df)


# In[264]:


type_grouped = pokemon_df[pokemon_df['Type 2']!='NA'].groupby(['Type 1', 'Type 2']).size()
print(type_grouped)


# In[265]:


sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(
    type_grouped.unstack(),
    annot=True,
)
plt.xticks(rotation=90)
plt.show()


# In[266]:


legendry_df = pokemon_df[pokemon_df['Legendary']==True]


# In[267]:


type_grouped = legendry_df[legendry_df['Type 2']!='NA'].groupby(['Type 1', 'Type 2']).size()
sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(
    type_grouped.unstack(),
    annot=True,
)
plt.xticks(rotation=90)
plt.show()


# In[268]:


pokemon_gen = legendry_df.groupby('Generation')['Name'].count()
sns.lineplot(data=pokemon_gen)


# In[269]:


max_type1_per_gen = pokemon_df.groupby(['Generation','Type 1']).size()


# In[270]:


max_type1_per_gen.unstack().plot()


# # We will use Bokeh library for Drawing interactive plot

# Code modified from : https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/legends.html

# In[271]:


type1_per_gen = pd.DataFrame({'count' : pokemon_df.groupby( [ "Generation", "Type 1"] ).size()}).reset_index()
print(pokemon_df.groupby( [ "Generation", "Type 1"] ).size())


# In[272]:


from bokeh.palettes import Spectral11
from bokeh.plotting import figure, output_file, show
from bokeh.models import Legend, LegendItem
p = figure(plot_width=800, plot_height=550, x_range=(1, 7))
p.background_fill_color = "beige"

p.title.text = 'Click on legend entries to hide the corresponding lines'
import random
legend_list = []
for type_id in type1_per_gen['Type 1'].unique():
    color = random.choice(Spectral11)
    df = pd.DataFrame(type1_per_gen[type1_per_gen['Type 1']==type_id])
    p.line(df['Generation'], df['count'], line_width=2, alpha=0.8, color=color, legend=type_id)

p.legend.location = "top_right"
p.legend.click_policy="hide"

show(p)


# In[273]:


pokemon_df.groupby([ "Generation", "Type 1"])[['Total']].max()


# In[274]:


type1_total_gen = pd.DataFrame({'Total' : pokemon_df.groupby( [ "Generation", "Type 1"] )['Total'].max()}).reset_index()
print(pokemon_df.groupby( [ "Generation", "Type 1"] )['Total'].max())


# In[275]:


from bokeh.palettes import Spectral11
from bokeh.plotting import figure, output_file, show
from bokeh.models import Legend, LegendItem
p = figure(plot_width=800, plot_height=550, x_range=(1, 7))
p.background_fill_color = "beige"

p.title.text = 'Click on legend entries to hide the corresponding lines'
import random
legend_list = []
for type_id in type1_total_gen['Type 1'].unique():
    color = random.choice(Spectral11)
    df = pd.DataFrame(type1_total_gen[type1_total_gen['Type 1']==type_id])
    p.line(df['Generation'], df['Total'], line_width=2, alpha=0.8, color=color, legend=type_id)

p.legend.location = "top_right"
p.legend.click_policy="hide"

show(p)


# # Red Wine Data Analysis

# All to together

# In[276]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[277]:


red_wine_df = pd.read_csv('C:\\Users\\arulkumar\\Desktop\\winequality-red.csv', delimiter=';')


# In[278]:


red_wine_df.head()


# In[279]:


red_wine_df.columns


# Descriptive Statistics

# In[280]:


red_wine_df.info()


# In[281]:


red_wine_df.describe()


# In[282]:


red_wine_df['quality'].head()


# Since there are no null entries, we don't need to deal with missing values.
# 

# In[283]:


##Analysis over Red Wine


# Let's first check the Quality Column
# 

# In[284]:


sns.set(rc={'figure.figsize':(7,6)})
sns.countplot(red_wine_df['quality'])


# Lets check which of the other columns are highly correlated to Quality
# 

# In[285]:


sns.pairplot(red_wine_df)


# In[286]:


sns.set(rc={'figure.figsize':(12,10)})
sns.heatmap(red_wine_df.corr(), annot=True, fmt='.2f', linewidths=2)


# •	Free Suplhur Dioxide and Total Sulphur Dioxide have some positive relation to Residual Sugar. On further inspection, I found that the quantity of SO2 is dependent on Sugar content. Reference : http://thewinehub.com/home/2013/01/09/the-use-or-not-of-sulfur-dioxide-in-winemaking-trick-or-treat/ . More specifically, the mentioned link states that "the lower the Residual Sugar , the less SO2 needed"
# 
# •	Density has a postive correlation with fixed acidity and residual sugar
# 
# •	Density has negative correlation with alcohol and pH
# 
# •	Quality has positive correlation with alcohol, citric acid and sulphates, and -ve correlation with citric acid. We need to explore this further.
# 
# •	Fixed acidity has high +ve correlation with citric acid and density and -ve correlation with pH
# 
# •	Residual sugar has +ve correlation with citric acid
# 
# •	pH has -ve correlation with fixed acidity and citric acid, but +ve correlation with volatile acid
# 

# In[287]:


sns.distplot(red_wine_df['alcohol'])


# In[288]:


from scipy.stats import skew
skew(red_wine_df['alcohol'])


# # Alcohol content is positively skewed
# 

# In[289]:


def draw_hist(temp_df, bin_size = 15):
    ax = sns.distplot(temp_df)
    #xmin, xmax = ax.get_xlim()
    #ax.set_xticks(np.round(np.linspace(xmin, xmax, bin_size), 2))
    plt.tight_layout()
    plt.locator_params(axis='y', nbins=6)
    plt.show()
    print("Skewness is {}".format(skew(temp_df)))
    print("Mean is {}".format(np.median(temp_df)))
    print("Median is {}".format(np.mean(temp_df)))


# In[290]:


draw_hist(red_wine_df['alcohol'])


# Let's see how alcohol varies w.r.t. qualit# 

# In[291]:


sns.boxplot(x='quality', y='alcohol', data=red_wine_df)


# In[292]:


sns.boxplot(x='quality', y='alcohol', data=red_wine_df,
           showfliers=False)


# In[293]:


joint_plt = sns.jointplot(x='alcohol', y='pH', data=red_wine_df,
                        kind='reg')


# In[294]:


from scipy.stats import pearsonr
def get_corr(col1, col2, temp_df):
    pearson_corr, p_value = pearsonr(temp_df[col1], temp_df[col2])
    print("Correlation between {} and {} is {}".format(col1, col2, pearson_corr))
    print("P-value of this correlation is {}".format(p_value))


# In[295]:


get_corr('alcohol', 'pH', red_wine_df)


# In[296]:


joint_plt = sns.jointplot(x='alcohol', y='density', data=red_wine_df,
                        kind='reg')


# In[297]:


get_corr('alcohol', 'density', red_wine_df)


# In[298]:


g = sns.FacetGrid(red_wine_df, col="quality")
g = g.map(sns.regplot, "density", "alcohol")


# Lets analyze sulphates and quality
# 

# In[303]:


sns.boxplot(x='quality', y='sulphates', data=red_wine_df,showfilers=False)


# In[302]:


sns.boxplot(x='quality', y='total sulfur dioxide', data=red_wine_df,showfliers=False)


# In[305]:


sns.boxplot(x='quality', y='free sulfur dioxide', data=red_wine_df,showfliers=False)


# In[306]:


red_wine_df.columns


# Lets move on to fixed acidity, volatile acidity and citric acid

# In[307]:


sns.boxplot(x='quality', y='fixed acidity', data=red_wine_df)


# In[308]:


sns.boxplot(x='quality', y='citric acid', data=red_wine_df)


# In[309]:


sns.boxplot(x='quality', y='volatile acidity', data=red_wine_df)


# # Trends between other columns
# 

# In[310]:


red_wine_df.columns


# In[311]:


get_corr('pH', 'citric acid', red_wine_df)


# # Create a new Column Total Acidity

# In[312]:


red_wine_df['total acidity'] = (red_wine_df['fixed acidity']+ red_wine_df['citric acid'] + red_wine_df['volatile acidity'])
sns.boxplot(x='quality', y='total acidity', data=red_wine_df,
           showfliers=False)


# In[313]:


sns.regplot(x='pH', y='total acidity', data=red_wine_df)


# In[314]:


g = sns.FacetGrid(red_wine_df, col="quality")
g = g.map(sns.regplot, "total acidity", "pH")


# In[315]:


get_corr('total acidity', 'pH', red_wine_df)


# In[316]:


g = sns.FacetGrid(red_wine_df, col="quality")
g = g.map(sns.regplot, "free sulfur dioxide", "pH")


# # White Wine Analysis

# In[317]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[319]:


white_wine_df = pd.read_csv("C:\\Users\\arulkumar\\Desktop\\winequality-white.csv", delimiter=';')


# In[320]:


white_wine_df.head()


# # White Wine Vs Red Wine Comparion

# In[321]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[322]:


red_wine_df = pd.read_csv("C:\\Users\\arulkumar\\Desktop\\winequality-red.csv", delimiter=';')
white_wine_df = pd.read_csv("C:\\Users\\arulkumar\\Desktop\\winequality-white.csv", delimiter=';')


# In[323]:


red_wine_df.head()


# In[324]:


white_wine_df.head()


# In[326]:


red_wine_df.info()


# In[328]:


white_wine_df.info()


# In[330]:


red_wine_df.describe()


# In[331]:


white_wine_df.describe()


# In[332]:


sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(white_wine_df.corr(), annot=True, fmt='.2f', linewidth=2)


# In[333]:


sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(red_wine_df.corr(), annot=True, fmt='.2f', linewidth=2)


# 	Red Wine	White Wine
# Quality	+ve Correlation :	
# 	- Alcohol	+ve Correlation :
# 	- Fixed Acidity	- Alcohol
# 	- sulphates	- pH (Weak)
# 	- Citric Acid	
# 		
# 	- -ve Correlation :	
# 	- Volatile Acidity	
# 	- Total Sulfur dioxide	
# 		- -ve Correlation :
# 	- density	
# 	- chlorides	- Volatile Acidity
# 		- chlorides
# 		- Total Sulfur dioxide
# 		
# 		- density
# 		- residual sugar(weak)
# ![image.png](attachment:image.png)

# # Combine the datasets

# In[336]:


red_wine_df['type'] = 'Red'
white_wine_df['type'] = 'White'


# In[342]:


wines_df = pd.concat([red_wine_df,white_wine_df])


# In[338]:


wines_df.head()


# In[339]:


wines_df.describe()


# In[343]:


wines_df.info()


# # Comparitive Analysis

# In[345]:


sns.countplot(x='quality', hue='type', data=wines_df)


# In[346]:


sns.set(rc={'figure.figsize':(10,10)})
p1=sns.kdeplot(red_wine_df['quality'], shade=True, color = "r", label="red wine")
p1=sns.kdeplot(white_wine_df['quality'], shade=True, color="b", label="white wine")


# In[347]:


sns.boxplot(x='quality',y='alcohol', hue='type', data=wines_df, palette=["r", "w"])


# In[349]:


sns.boxplot(x='quality',y='alcohol', hue='type', data=wines_df, palette=["r", "w"],showfliers=False)


# In[351]:


sns.boxplot(x='quality',y='density', hue='type', data=wines_df, palette=["r", "w"])


# In[352]:


sns.boxplot(x='quality',y='density', hue='type', data=wines_df,
            palette=["r", "w"], showfliers=False)


# In[353]:


sns.jointplot(x='alcohol', y='residual sugar', data=red_wine_df)


# In[354]:


sns.boxplot(x='quality',y='residual sugar', hue='type', 
            data=wines_df, palette=["r", "w"], showfliers=False)


# In[355]:


sns.set(rc={'figure.figsize':(10,10)})
p1=sns.kdeplot(red_wine_df['residual sugar'], shade=True, color = "r", label="red wine")
p1=sns.kdeplot(white_wine_df['residual sugar'], shade=True, color="b", label="white wine")


# In[356]:


sns.regplot(x='alcohol', y='residual sugar', data=white_wine_df)


# In[357]:


sns.boxplot(x='quality',y='total sulfur dioxide', hue='type', data=wines_df, palette=["r", "w"])


# In[358]:


sns.boxplot(x='quality',y='free sulfur dioxide', hue='type', 
            data=wines_df, palette=["r", "w"], showfliers=False)


# In[359]:


sns.boxplot(x='quality',y='sulphates', hue='type', 
            data=wines_df, palette=["r", "w"], showfliers=False)


# In[360]:


sns.boxplot(x='quality',y='citric acid', hue='type', data=wines_df,
            palette=["r", "w"], showfliers=False)


# In[361]:


sns.boxplot(x='quality',y='chlorides', hue='type', data=wines_df,
            palette=["r", "w"], showfliers=False)


# In[362]:


wines_df['total acidity'] = wines_df['fixed acidity'] + wines_df['volatile acidity'] + wines_df['citric acid']
sns.boxplot(x='quality',y='total acidity', hue='type', data=wines_df,
            palette=["r", "w"], showfliers=False)


# In[ ]:




