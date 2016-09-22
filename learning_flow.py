
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


from os.path import expanduser
try:
   import cPickle as pickle
except:
   import pickle

PLACEHOLDER = 0

small_dataset_size = 5000
test_set_size = 1000


home = expanduser("~")
data_per_vessel = pickle.load (open( home+"\\Dropbox\\Sea Snails\\preproc data\\data_per_vessel_lists.p", "rb" ) )
meetings_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\meetings_train.p", "rb" ) )
port_visits_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\port_visits_train_raw.p", "rb" ) )
vessels_labels_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train_raw.p", "rb" ) )


def meeting_feature(args):
    meetings = meetings_train_raw.loc[args['meeting_id1_row']]
    avg = PLACEHOLDER if meetings.empty else meetings['duration_min'].mean()
    num = len(args['meeting_id1_row']) + len(args['meeting_id1_row'])
    return pd.Series([avg,num], index=['avg', 'num'])

small_data = data_per_vessel[1:small_dataset_size]
feature_table = data_per_vessel[1:small_dataset_size].apply(meeting_feature, axis=1)
train_set = feature_table[1:-test_set_size]
train_labels = small_data['type'].iloc[1:-test_set_size]

le = preprocessing.LabelEncoder()
le.fit(train_labels)
encoded_train_labels = le.transform(train_labels)

test_set = feature_table[-test_set_size:].as_matrix()
test_labels = small_data['type'].iloc[-test_set_size:]
encoded_test_labels = le.transform(test_labels)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_set, encoded_train_labels)

result  =  neigh.predict(test_set)

error_vector = result == encoded_test_labels

print ('Accuracy:', np.sum(error_vector) / len(error_vector))












