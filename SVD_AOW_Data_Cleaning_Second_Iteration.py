
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVDpp


# In[2]:


AOW_test = "../../Take/test_final.csv"
AOW_train = "../../Take/Train_AOW.csv"


# In[3]:


dict_test = pickle.load(open(AOW_test, "br"))


# In[4]:


array_test = [ [k, v[0], v[1]] for k in dict_test.keys() for v in dict_test[k][0] ]


# In[5]:


df_test = pd.DataFrame(data=array_test, columns=["user", "movie", "interaction"])


# In[6]:


df_train = pd.read_csv(AOW_train, sep=";", index_col=0, header=0)


# In[7]:


reader = Reader(rating_scale=(1, 1))


# In[8]:


data = Dataset.load_from_df(df_train, reader)


# In[11]:


svdpp = SVDpp()


# In[ ]:


vld = cross_validate(svdpp, data)

