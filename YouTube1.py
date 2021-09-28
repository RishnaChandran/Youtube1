#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install wordcloud')


# In[2]:


#4.Approach for Data cleansing 
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
#Importing Libraries
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import cm
from datetime import datetime
import glob
import os
import json
import pickle
import six
sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None


# In[3]:


#Importing all the CSV files
allcsv=[i for i in glob.glob('*.{}'.format('csv'))]
allcsv


# In[4]:


#Reading all CSV files

all_dfs = [] # list to store each data frame separately
for csv in allcsv:
    df = pd.read_csv(csv)
    df['country'] = csv[0:2] # adding column 'country' so that each dataset could be identified uniquely
    all_dfs.append(df)
all_dfs[0].head() # index 0 to 9 for [CA, DE, FR, GB, IN, JP, KR, MX, RU, US] datasets


# In[5]:


#Number of rows and columns dataset contains
df.shape


# In[6]:


#summary of columns and their associated data types 
df.info()


# In[7]:


#Statistical summary of our numerical columns
df.describe()


# In[8]:


#Statistical summary of our categorical columns
#Publish Time column with data type : Object
df.describe(include=['O'])


# In[9]:


#Fixing Data Types

for df in all_dfs:
    # video_id 
    df['video_id'] = df['video_id'].astype('str') 
    # trending date
    df['trending_date'] = ['20'] + df['trending_date']
    df['trending_date'] = pd.to_datetime(df['trending_date'], format = "%Y.%d.%m")
    #title
    df['title'] = df['title'].astype('str')
    #channel_title
    df['channel_title'] = df['channel_title'].astype('str')
    #category_id
    df['category_id'] = df['category_id'].astype(str) 
    #tags
    df['tags'] = df['tags'].astype('str')
    
    # views, likes, dislikes, comment_count are already in correct data types i.e int64
    
    #thumbnail_link
    df['thumbnail_link'] = df['thumbnail_link'].astype('str') 
    #description
    df['description'] = df['description'].astype('str')
    # Changing comments_disabled, ratings_disabled, video_error_or_removed from bool to categorical
    df['comments_disabled'] = df['comments_disabled'].astype('category') 
    df['ratings_disabled'] = df['ratings_disabled'].astype('category') 
    df['video_error_or_removed'] = df['video_error_or_removed'].astype('category') 
    
    # publish_time 
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')


# In[10]:


df['publish_time'].head()


# In[11]:


df['trending_date'].head()


# In[15]:


all_dfs[1].dtypes


# In[18]:


# Changing data type for 'publish_date' from object to 'datetime64[ns]'
for df in all_dfs:
    df.insert(4, 'publish_date', df['publish_time'].dt.date) # loc, column name, values for column to be inserted
    df['publish_time'] = df['publish_time'].dt.time
# Changing data type for 'publish_date' from object to 'datetime64[ns]'
for df in all_dfs:
     df['publish_date'] = pd.to_datetime(df['publish_date'], format = "%Y-%m-%d")


# In[19]:


all_dfs[1].dtypes


# In[20]:


df.head()


# In[22]:


#Examining Missing Values
for df in all_dfs:
    sns.heatmap(df.isnull(), cbar=False)
    plt.figure()


# In[23]:


#Combining Every Dataframe Into One Huge Dataframe
combined_df = pd.concat(all_dfs)


# In[24]:


# Making copy of original dataframe
backup_df = combined_df.reset_index().sort_values('trending_date', ascending=False).set_index('video_id')
# Sorting according to latest trending date while removing duplicates
combined_df = combined_df.reset_index().sort_values('trending_date', ascending=False).drop_duplicates('video_id',keep='first').set_index('video_id')
# Doing the same above operation for each of the individual dataframes in the list we created earlier
for df in all_dfs:
    df = df.reset_index().sort_values('trending_date', ascending=False).set_index('video_id')
# Printing results
combined_df[['publish_date','publish_time','trending_date', 'country']].head()
# It can be seen that latest publications and trending information is at the top now


# In[25]:


#Inserting Category Column 
#One of our final steps for the data cleaning of the data sets was checking the JSON files that were available with the data sets
#Read file
with open('US_category_id.json', 'r') as f:  # reading one randomly selected json files to make sense of its contents
    data = f.read()
# parse file
obj = json.loads(data)
# printing
obj


# In[26]:


category_id = {}
with open('DE_category_id.json', 'r') as f:
    d = json.load(f)
    for category in d['items']:
        category_id[category['id']] = category['snippet']['title']
combined_df.insert(2, 'category', combined_df['category_id'].map(category_id))
backup_df.insert(2, 'category', backup_df['category_id'].map(category_id))
for df in all_dfs:
    df.insert(2, 'category', df['category_id'].map(category_id))
# Printing cleaned combined dataframe
combined_df.head(3)


# In[27]:


combined_df['category'].unique()


# In[28]:


#5.Figure out if there is a correlation between the number of views and the number of likes.
columns_of_interest = ['views', 'likes']
corr_matrix = df[columns_of_interest].corr()
corr_matrix


# In[29]:


fig, ax = plt.subplots()
heatmap = ax.imshow(corr_matrix, interpolation='nearest', cmap=cm.coolwarm)
# making the colorbar on the side
cbar_min = corr_matrix.min().min()
cbar_max = corr_matrix.max().max()
cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])
# making the labels
labels = ['']
for column in columns_of_interest:
    labels.append(column)
    labels.append('')
ax.set_yticklabels(labels, minor=False)
ax.set_xticklabels(labels, minor=False)
plt.show()


# In[30]:


#6.List top trending videos in each category 
# Getting names of all countries
countries = []
allcsv = [i for i in glob.glob('*.{}'.format('csv'))]
for csv in allcsv:
    c = csv[0:2]
    countries.append(c)
for country in countries:
    if country == 'US':
        tempdf = combined_df[combined_df['country']==country]['category'].value_counts().reset_index()
        ax = sns.barplot(y=tempdf['index'], x=tempdf['category'], data=tempdf, orient='h')
        plt.xlabel("Number of Videos")
        plt.ylabel("Categories")
        plt.title("Catogories of trend videos in " + country)
    else:
        tempdf = combined_df[combined_df['country']==country]['category'].value_counts().reset_index()
        ax = sns.barplot(y=tempdf['index'], x=tempdf['category'], data=tempdf, orient='h')
        plt.xlabel("Number of Videos")
        plt.ylabel("Categories")
        plt.title("Catogories of trend videos in " + country)
        plt.figure()


# In[35]:


#7.Create a new feature called total_trend_days that just calculates the number of days a video has trended in a new column.
# Calculating days between publish and trending date
temporary = []
for total_trend_days in all_dfs:
    temp = total_trend_days
    temp['total_trend_days'] = (temp['trending_date'] - temp['publish_date']).dt.days
    temporary.append(temp)
total_trend_days.sort_values(by='total_trend_days',ascending=False).head()


# In[33]:


#Remove irrelevant columns.
del_cols = ['thumbnail_link','comments_disabled','ratings_disabled','video_error_or_removed','description']
df=df.drop(del_cols, axis=1)


# In[36]:


#8.Figure out the relationship between the published_time and trending_date
columns_of_interest = ['publish_time', 'trending_date']
corr_matrix = df[columns_of_interest].corr("pearson")
corr_matrix


# In[37]:


temp = combined_df
temp = temp.groupby('category')['views', 'likes'].apply(lambda x: x.astype(int).sum())
temp = temp.sort_values(by='likes', ascending=False).head()
temp


# In[43]:


temp = combined_df
temp = temp.groupby('category')['views','likes','publish_time','trending_date'].apply(lambda x: x.astype(str).sum())
temp = temp.sort_values(by='likes', ascending=False).head()
temp


# In[44]:


#9.Identify non-date format values in a trending_date column
df['trending_date']= pd.to_datetime('13000101', format='%Y%m%d', errors='coerce')
df['trending_date'].head(10)


# In[45]:


df['trending_date']= pd.to_datetime(df['trending_date'], format='%Y%m%d', errors='coerce')
df['trending_date'].head(10)


# In[ ]:




