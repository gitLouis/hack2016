

import pandas
import networkx
from datetime import datetime, timedelta
import numpy

from geopy.distance import great_circle
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
meetings_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\tmp_meetings_train.p", "rb" ) )
port_visits_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\port_visits_train_raw.p", "rb" ) )
vessels_labels_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train_raw.p", "rb" ) )
meetings_train_raw['port_id'] = meetings_train_raw['Lat'].apply(lambda x: str(int(round(x)))) + meetings_train_raw['Long'].apply(    lambda x: str(int(round(x))))
pickle.dump( meetings_train_raw, open( home+"\\Dropbox\\Sea Snails\\raw data\\meetings_train.p", "wb" ) )