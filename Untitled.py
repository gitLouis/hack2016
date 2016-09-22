
# coding: utf-8

import pandas
from os.path import expanduser
try:
   import cPickle as pickle
except:
   import pickle


home = expanduser("~")

# load raw data
# print('loading data from CSV...')
# meetings_train_raw = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\raw data\\meetings_train.csv", nrows = None)
# port_visits_train_raw = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\raw data\\port_visits_train.csv", nrows = None)
# vessels_labels_train_raw = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train.csv", index_col = 'vessel_id')

# print('saving to pickle file...')
# pickle.dump( meetings_train_raw, open( home+"\\Dropbox\\Sea Snails\\raw data\\meetings_train.p", "wb" ) )
# pickle.dump( port_visits_train_raw, open( home+"\\Dropbox\\Sea Snails\\raw data\\port_visits_train_raw.p", "wb" ) )
# pickle.dump( vessels_labels_train_raw, open( home+"\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train_raw.p", "wb" ) )

print('loading data from pickle file...')
meetings_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\meetings_train.p", "rb" ) )
port_visits_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\port_visits_train_raw.p", "rb" ) )
vessels_labels_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train_raw.p", "rb" ) )

#create sample data
# meetings_train_raw = meetings_train_raw[0:1000]
# port_visits_train_raw = port_visits_train_raw[0:1000]

# create list of index of vessles to records in 'meetings' and in 'port visits' by row number
print('creating vessle data...')
data_per_vessel = vessels_labels_train_raw.copy()
print('creating vessle data...meetings 1...')

data_per_vessel['meeting_id1_row'] = meetings_train_raw.reset_index().groupby('ves_id1').index.aggregate(lambda x: tuple(x))
print('creating vessle data...meetings 2...')
data_per_vessel['meeting_id2_row'] = meetings_train_raw.reset_index().groupby('ves_id2').index.aggregate(lambda x: tuple(x))
print('creating vessle data...port visit...')
data_per_vessel['port_visits_row'] = port_visits_train_raw.reset_index().groupby('ves_id').index.aggregate(lambda x: tuple(x))

# save vessle data to pickle
pickle.dump( data_per_vessel, open( home+"\\Dropbox\\Sea Snails\\preproc data\\data_per_vessel.p", "wb" ) )

# add features
print('adding features...')
data_per_vessel['num_of_meetings'] = len(data_per_vessel['meeting_id1_row'])+len(data_per_vessel['meeting_id2_row'])

meetings_train_sample = meetings_train_raw[0:100]
port_visits_train_sample = port_visits_train_raw[0:100]


c = pandas.Series.unique(meetings_train_sample['ves_id1'])



labels = vessels_labels_train_raw.loc[c]


print(labels[0:5])


# In[111]:

meeting_arr = labels.apply(lambda x: list(meetings_train_raw.loc[(meetings_train_raw['ves_id1']==x[0]) | (meetings_train_raw['ves_id2']==x[0])]))


# In[112]:

meeting_arr.head()


# In[87]:

labels.head(1)



print(labels)

len(labels)


# In[26]:




# In[78]:

type(meeting_arr)


# In[25]:

result = c.join(vessels_labels_train_raw, on='ves_id1')


# In[14]:

print(meetings_train_raw[0:5])


# In[19]:

print(vessels_labels_train_raw[0:5])


# In[ ]:



