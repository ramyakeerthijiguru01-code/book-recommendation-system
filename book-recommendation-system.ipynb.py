#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[74]:


df_user = pd.read_csv('BX-Users.csv',encoding='latin-1')


# In[75]:


df_user.head()


# In[76]:


df_user.isnull().sum()


# In[77]:


sns.histplot(df_user['Age'])


# In[78]:


df_user['Age'].fillna(df_user['Age'].mean(),inplace=True)


# In[79]:


df_user.isnull().sum()


# In[80]:


df_user['Location'].fillna('NA',inplace=True)


# In[81]:


df_user.isnull().sum()


# In[82]:


#Read the books data and explore


# In[83]:


#columns_names = ['isbn','book_title']
df_books = pd.read_csv('BX-Books.csv', encoding='latin-1')


# In[84]:


df_books.head()


# In[85]:


df_books.isnull().sum()


# In[86]:


df_books.dropna(inplace=True)


# Now Read the data Ratings are given by user. you will read only first 10K rows otherwise, out of memory error can occur

# In[87]:


df = pd.read_csv('BX-Book-Ratings.csv',encoding='latin-1',nrows=10000)


# In[88]:


df.head()


# In[89]:


df.describe()


# In[90]:


df.isnull().sum()


# Merge the Dataframes. For all practical purpose, user Master Data is not required, so ignore dataframe df_user

# In[91]:


df = pd.merge(df,df_books,on='isbn')


# In[92]:


df.head()


# first problem statement
# 1) number of unique users and books

# Now lets take a quick look at the number of unique users and books

# In[93]:


n_users = df.user_id.nunique()
n_books = df.isbn.nunique()

print('Num of Users: '+ str(n_users))
print('Num of Books: '+str(n_books))


# Convert ISBN to numeruic number in order

# In[94]:


df.dtypes


# In[95]:


isbn_list = df.isbn.unique()
print(" Length of isbn List:", len(isbn_list))


# In[96]:


def get_isbn_numeric_id(isbn):
    #print (" isbn is:", isbn)
    itemindex = np.where(isbn_list==isbn)
    return itemindex[0][0]


# In[97]:


df['isbn_id'] = df['isbn'].apply(get_isbn_numeric_id)


# In[98]:


df.head(20)


# In[99]:


df.shape


# second problem statement
# 
# 2)Do the same for user_id, convert it into numeric data in order

# In[100]:


userid_list = df.user_id.unique()
print(" Length of user_id List:", len(userid_list))
def get_user_id_numeric_id(user_id):
    #print (" isbn is:", isbn)
    itemindex = np.where(userid_list==user_id)
    return itemindex[0][0]


# converting both user_id and isbn to ordered list i.e, from 0 to n-1

# In[101]:


df['user_id_order'] = df['user_id'].apply(get_user_id_numeric_id)


# In[102]:


df.head()


# 3)Re-index the columns to build a matrix

# In[103]:


new_col_order = ['user_id_order', 'isbn_id', 'rating', 'book_title', 'book_author', 'year_of_publication', 'publisher', 'isbn', 'user_id']
df =df.reindex(columns=new_col_order)
df.head()


# In[104]:


df.tail()


# Train Test Split

# Recommendation system is difficult to evcaluate, but you will still learn how to eveluate them.
# In order to do this,
# you will split your data into two sets.However, you wont do your classic x_train,x_test,y_train,y_test split. 
# Instead, you can actually just segement the data into two sets odf data

# In[105]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.30)


# In[106]:


train_data.shape


# In[107]:


test_data.shape


# In[108]:


train_data


# In[109]:


test_data


# Approach: You will use memory_based collaborative filtering
# Memory based collabarative filtering approches can be divided into two main selections : User_Item filtering and item_item filtering
# 
# As User_item filtering will take prticular user, find users that are similar ti that user based on similarity of ratings, and 
# recommended items that those similar users liked
# 
# 
# 

# Now we are going to create train_data matrix and test data matrix

# In[111]:


train_data_matrix = np.zeros((n_users, n_books))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3] 
train_data_matrix.shape


# In[112]:


train_data_matrix.shape


# In[113]:


train_data_matrix


# In[114]:


test_data_matrix = np.zeros((n_users, n_books))
for line in test_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3] 
test_data_matrix.shape


# In[115]:


test_data_matrix.shape


# In[116]:


test_data_matrix


# You can use the pairwise_distance function from sklearn to calculate the cosine similarity.note, the output will
# range from 0 to 1 since the ratings are all positive
# 

# In[117]:


from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
#T means Transpose the data


# In[118]:


user_similarity


# In[119]:


item_similarity


# Next step is to make predictions

# Formula
# p= mean(rating)+difference rating*Similarity/ Sum of all similarity

# In[120]:


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


# In[121]:


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# In[122]:


item_prediction


# In[123]:


user_prediction


# Evaluation
# 
# There are many evaluation metrics, but one of the most popular metric used to evaluate accuracy of predicted ratings ROOT MEAN SQUARE
# ERROR (RMSE)
# 
# Since you oly want to consider predicted ratings that are in the test dataset, you filter out all other elements in the prediction matrix with:
# prediction[growth_truth.nonzero()]   

# In[124]:


from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# In[129]:


print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))


# Both the approch yiels almost same result
