
# coding: utf-8

import pandas

meetings_train_raw = pandas.read_csv("C:\\Users\\Maria\\Dropbox\\Sea Snails\\raw data\\meetings_train.csv")
port_visits_train_raw = pandas.read_csv("C:\\Users\\Maria\\Dropbox\\Sea Snails\\raw data\\port_visits_train.csv")
vessels_labels_train_raw = pandas.read_csv("C:\\Users\\Maria\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train.csv", index_col = 'vessel_id')

port_visits_train_raw.head()


vessels_labels_train_raw.apply(g)

f = vessels_labels_train_raw.set_index('vessel_id')

f['meeting_id1_row'] = meetings_train_raw.reset_index().groupby('ves_id1').index.aggregate(lambda x: tuple(x)).to_frame()
f['meeting_id2_row'] = meetings_train_raw.reset_index().groupby('ves_id2').index.aggregate(lambda x: tuple(x))
f['port_visits_row'] = port_visits_train_raw.reset_index().groupby('ves_id').index.aggregate(lambda x: tuple(x))
f.reset_index().count()


# In[131]:

f.head()


# In[109]:

g.first()


# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:

meetings_train_sample = meetings_train_raw[0:100]
port_visits_train_sample = port_visits_train_raw[0:100]


# In[63]:

c = pandas.Series.unique(meetings_train_sample['ves_id1'])


# In[64]:

labels = vessels_labels_train_raw.loc[c]


# In[85]:

print(labels[0:5])


# In[111]:

meeting_arr = labels.apply(lambda x: list(meetings_train_raw.loc[(meetings_train_raw['ves_id1']==x[0]) | (meetings_train_raw['ves_id2']==x[0])]))


# In[112]:

meeting_arr.head()


# In[87]:

labels.head(1)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

final_table


# In[38]:

print(labels)


# In[ ]:




# In[68]:

len(labels)


# In[26]:




# In[78]:

type(meeting_arr)


# In[25]:

result = c.join(vessels_labels_train_raw, on='ves_id1')


# In[ ]:




# In[14]:

print(meetings_train_raw[0:5])


# In[19]:

print(vessels_labels_train_raw[0:5])


# In[ ]:



