
import pandas
from os.path import expanduser
try:
   import cPickle as pickle
except:
   import pickle


home = expanduser("~")

meetings_test_raw = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\test data\\meetings_test.csv", nrows = None)
port_visits_test_raw = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\test data\\port_visits_test.csv", nrows = None)
vessels_labels_test_raw = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\test data\\vessels_to_label1.csv", index_col = 'vessel_id')
meetings_test_raw['port_id'] = meetings_test_raw['Lat'].apply(lambda x: str(int(round(x)))) + meetings_test_raw['Long'].apply(    lambda x: str(int(round(x))))

print('saving to pickle file...')
pickle.dump( meetings_test_raw, open( home+"\\Dropbox\\Sea Snails\\test data\\meetings_test.p", "wb" ) )
pickle.dump( port_visits_test_raw, open( home+"\\Dropbox\\Sea Snails\\test data\\port_visits_test.p", "wb" ) )
pickle.dump( vessels_labels_test_raw, open( home+"\\Dropbox\\Sea Snails\\test data\\vessels_to_label.p", "wb" ) )

#create list of index of vessles to records in 'meetings' and in 'port visits' by row number
print('creating vessle data...')
data_per_vessel = vessels_labels_test_raw.copy()
print('creating vessle data...meetings 1...')

data_per_vessel['meeting_id1_row'] = meetings_test_raw.reset_index().groupby('ves_id1').index.aggregate(lambda x: tuple(x))
print('creating vessle data...meetings 2...')
data_per_vessel['meeting_id2_row'] = meetings_test_raw.reset_index().groupby('ves_id2').index.aggregate(lambda x: tuple(x))
print('creating vessle data...port visit...')
data_per_vessel['port_visits_row'] = port_visits_test_raw.reset_index().groupby('ves_id').index.aggregate(lambda x: tuple(x))

#save vessle data to pickle
pickle.dump( data_per_vessel, open( home+"\\Dropbox\\Sea Snails\\preproc data\\test_data_per_vessel.p", "wb" ) )

