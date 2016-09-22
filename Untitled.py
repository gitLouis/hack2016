
# coding: utf-8

import pandas

meetings_train_raw = pandas.read_csv("C:\\Users\\Maria\\Dropbox\\Sea Snails\\raw data\\meetings_train.csv")
port_visits_train_raw = pandas.read_csv("C:\\Users\\Maria\\Dropbox\\Sea Snails\\raw data\\port_visits_train.csv")
vessels_labels_train_raw = pandas.read_csv("C:\\Users\\Maria\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train.csv", index_col = 'vessel_id')

f = vessels_labels_train_raw.copy()

f['meeting_id1_row'] = meetings_train_raw.reset_index().groupby('ves_id1').index.aggregate(lambda x: tuple(x)).to_frame()
f['meeting_id2_row'] = meetings_train_raw.reset_index().groupby('ves_id2').index.aggregate(lambda x: tuple(x))
f['port_visits_row'] = port_visits_train_raw.reset_index().groupby('ves_id').index.aggregate(lambda x: tuple(x))
f.reset_index().count()

f.head()


