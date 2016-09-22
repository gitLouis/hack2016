
# coding: utf-8

import numpy
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

data_per_vessel['meeting_id1_row'] = meetings_train_raw.reset_index().groupby('ves_id1').index.aggregate(lambda x: list(x))
print('creating vessle data...meetings 2...')
data_per_vessel['meeting_id2_row'] = meetings_train_raw.reset_index().groupby('ves_id2').index.aggregate(lambda x: list(x))
print('creating vessle data...port visit...')
data_per_vessel['port_visits_row'] = port_visits_train_raw.reset_index().groupby('ves_id').index.aggregate(lambda x: list(x))

data_per_vessel['meeting_id1_row'] = data_per_vessel['meeting_id1_row'].apply(lambda x: [] if numpy.all(numpy.isnan(x)) else x )
data_per_vessel['meeting_id2_row'] = data_per_vessel['meeting_id2_row'].apply(lambda x: [] if numpy.all(numpy.isnan(x)) else x )


# save vessle data to pickle
pickle.dump( data_per_vessel, open( home+"\\Dropbox\\Sea Snails\\preproc data\\data_per_vessel_lists.p", "wb" ) )

# add features
print('adding features...')
data_per_vessel['num_of_meetings'] = len(data_per_vessel['meeting_id1_row'])+len(data_per_vessel['meeting_id2_row'])



