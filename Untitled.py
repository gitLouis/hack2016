
# coding: utf-8

import pandas
import networkx
from datetime import datetime, timedelta
import numpy

from geopy.distance import great_circle
from os.path import expanduser
import numpy.linalg as lin
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
#meetings_train_raw['port_id'] = meetings_train_raw['Lat'].apply(lambda x: str(int(round(x)))) + meetings_train_raw['Long'].apply(    lambda x: str(int(round(x))))

#create sample data
# meetings_train_raw = meetings_train_raw[0:1000]
# port_visits_train_raw = port_visits_train_raw[0:1000]

# create list of index of vessles to records in 'meetings' and in 'port visits' by row number
# print('creating vessle data...')
# data_per_vessel = vessels_labels_train_raw.copy()
# print('creating vessle data...meetings 1...')

# data_per_vessel['meeting_id1_row'] = meetings_train_raw.reset_index().groupby('ves_id1').index.aggregate(lambda x: tuple(x))
# print('creating vessle data...meetings 2...')
# data_per_vessel['meeting_id2_row'] = meetings_train_raw.reset_index().groupby('ves_id2').index.aggregate(lambda x: tuple(x))
# print('creating vessle data...port visit...')
# data_per_vessel['port_visits_row'] = port_visits_train_raw.reset_index().groupby('ves_id').index.aggregate(lambda x: tuple(x))

# save vessle data to pickle
# pickle.dump( data_per_vessel, open( home+"\\Dropbox\\Sea Snails\\preproc data\\data_per_vessel.p", "wb" ) )
data_per_vessel = pickle.load(  open( home+"\\Dropbox\\Sea Snails\\preproc data\\data_per_vessel_lists2.p", "rb" ) )


# def averageMoveDist( port_visits, meetings ):
#    # Add both the parameters and return them."
#    total = (len(x) if type(x) is tuple else 0) + (len(y) if type(y) is tuple else 0)
#    return total


def createGraph(ship_row):


    meetings = meetings_train_raw[['port_id','start_time','duration_min','Lat','Long']].loc[ship_row['port_visits_row']]
    meetings = meetings.dropna()
    if len(meetings)>0 :
        meetings['event_type'] = 'meeting'

    port_visits = port_visits_train_raw[['port_id','start_time','duration_min','Lat','Long']].loc[ship_row['meeting_id1_row']]
    port_visits = port_visits.dropna()
    if len(port_visits)>0 :
        port_visits['event_type'] = 'port_visit'
    ship_vertices = meetings.append(port_visits);
    ship_vertices = ship_vertices.sort_values('start_time')
    start_times = ship_vertices['start_time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    vertex_durations = ship_vertices['duration_min'].apply(lambda x: timedelta(minutes=x))
    end_times = start_times + vertex_durations
    edge_durations = [start_times.iloc[i+1] - end_times.iloc[i] for i in range(0,len(start_times)-1)]
    # list(end_times[1:])-list(start_times[0:-1])

    # great_circle((), ()).miles

    multiG = networkx.MultiDiGraph()
    multiG.add_nodes_from(port_visits_train_raw['port_id'].unique())
    multiG.add_nodes_from(meetings_train_raw['port_id'].unique())
    G = networkx.DiGraph()
    G.add_nodes_from(port_visits_train_raw['port_id'].unique())
    G.add_nodes_from(meetings_train_raw['port_id'].unique())

    for i in range(0,len(ship_vertices)-1):
        from_port = ship_vertices[['port_id']].iloc[[i]].values.tolist()[0][0]
        to_port =  ship_vertices[['port_id']].iloc[[i+1]].values.tolist()[0][0]
        edge_distance = int(great_circle(ship_vertices[['Lat','Long']].iloc[[i]].values.tolist()[0],ship_vertices[['Lat','Long']].iloc[[i+1]].values.tolist()[0]).kilometers)
        multiG.add_edge(from_port,to_port, duration=edge_durations[i],distance = edge_distance)
        if to_port in G.neighbors(from_port):
            cum_duration = G[from_port][to_port]['duration'] + edge_durations[i]
            cum_distance = G[from_port][to_port]['distance'] + edge_distance
            cum_count = G[from_port][to_port]['count'] + 1;
            G.add_edge(from_port, to_port, duration=cum_duration, distance=cum_distance, count = cum_count)
        else:
            G.add_edge(from_port, to_port, duration=edge_durations[i], distance=edge_distance, count=1)

    for u, v, d in G.edges(data=True):
        avg_duration = d['duration'] /d['count']
        avg_distance = d['distance'] /d['count']
        G.add_edge(from_port, to_port, duration=avg_duration, distance=avg_distance, count=cum_count)
    return multiG, G





# add features
print('adding features...')
# data_per_vessel['num_of_meetings'] = data_per_vessel[['meeting_id1_row','meeting_id2_row']].apply(lambda x: (len(x[1]) if type(x[1]) is tuple else 0) + (len(x[2]) if type(x[2]) is tuple else 0))

# t1 = data_per_vessel[['meeting_id1_row','meeting_id2_row']]
# data_per_vessel['num_of_meetings'] = data_per_vessel[['meeting_id1_row','meeting_id2_row']].apply(lambda x : countMeetings(x[[0]][0], x[[1]][0]),axis = 1 )

data_per_vessel['num_of_meetings'] = data_per_vessel['meeting_id1_row'].apply(len )
data_per_vessel['num_of_port_visits'] = data_per_vessel['port_visits_row'].apply(len )

ship_graphs = dict()

for index, row in data_per_vessel.iterrows():
    multiG, G = createGraph(row);

    multiG_numpy = networkx.to_numpy_matrix (multiG)
    P, D, Q = numpy.linalg.svd(multiG_numpy, full_matrices=False)
    d_multi = numpy.diag(D)

    multiG_numpy = networkx.to_numpy_matrix (G)
    P, D, Q = numpy.linalg.svd(multiG_numpy, full_matrices=False)
    d = numpy.diag(D)

    ship_graphs[index]


# data_per_vessel['num_of_port_visits'] = data_per_vessel['port_visits_row'].apply(lambda x: (len(x) if type(x) is tuple else 0))

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



