
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
from datetime import datetime, timedelta
from geopy.distance import great_circle
import time

from os.path import expanduser
try:
   import cPickle as pickle
except:
   import pickle

PLACEHOLDER = 0


home = expanduser("~")
data_per_vessel = pickle.load (open( home+"\\Dropbox\\Sea Snails\\preproc data\\data_per_vessel_lists2.p", "rb" ) )
meetings_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\meetings_train.p", "rb" ) )
port_visits_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\port_visits_train_raw.p", "rb" ) )
vessels_labels_train_raw = pickle.load( open( home+"\\Dropbox\\Sea Snails\\raw data\\vessels_labels_train_raw.p", "rb" ) )

ship_type_by_port_id = pickle.load( open( home+"\\Dropbox\\Sea Snails\\feature data\\ship_type_by_port_id_prob.p", "rb" ) )
ship_type_by_port_country = pickle.load( open( home+"\\Dropbox\\Sea Snails\\feature data\\ship_type_by_port_country.p", "rb" ) )

#graph_per_vessel = pickle.load

meeting_indicative_countries = pd.read_csv(home+"\\Dropbox\\Sea Snails\\country indicativeness\\meeting_country_indicativeness.csv", nrows = None)
port_indicative_countries = pd.read_csv(home+"\\Dropbox\\Sea Snails\\country indicativeness\\visiting_port_indicativeness.csv", nrows = None)
#meetings_classification = pd.read_csv(home+"\\Dropbox\\Sea Snails\\feature data\\.csv", nrows = None)


meeting_indicative_countries = list(meeting_indicative_countries['country'])
port_indicative_countries = list(port_indicative_countries['country'])



small_dataset_size = 1000
test_set_size = 100
validation_n = 10


#----------------------------Feature Repository-------------------------------------


def port_id_probs(args):
    port_visits = port_visits_train_raw.loc[args['port_visits_row']]
    port_probs = ship_type_by_port_id.loc[port_visits['port_id']]
    feature = port_probs.sum() / len(port_probs) if len(port_probs) >0  else  port_probs.sum()
    return feature

def country_id_probs(args):
    port_visits = port_visits_train_raw.loc[args['port_visits_row']]
    port_visits = port_visits[pd.notnull(port_visits['country'])]
    port_probs = ship_type_by_port_country.loc[port_visits['country']]
    feature = port_probs.sum() / len(port_probs) if len(port_probs) >0  else  port_probs.sum()
    return feature

def num_unique_ports(args):
    port_visits = port_visits_train_raw.loc[args['port_visits_row']]
    unique_ports = pd.Series.unique(port_visits['port_id'])
    feature = len(unique_ports) / len(port_visits) if len(port_visits) > 0 else 0
    return pd.Series([feature],index=['num_unique_ports'])

def num_country_visits(args):
    port_visits = port_visits_train_raw.loc[args['port_visits_row']]
    port_visits = port_visits[pd.notnull(port_visits['country'])]
    country = pd.get_dummies(port_visits['country'])
    return country.sum()

def main_country_weight(args):
    port_visits = port_visits_train_raw.loc[args['port_visits_row']]
    port_visits = port_visits[pd.notnull(port_visits['country'])]
    country = pd.get_dummies(port_visits['country'])
    feature = country.sum().max() / len(port_visits) if len(port_visits) > 0 else 0
    return pd.Series([feature],index=['main_country_weight'])

def main_port_weight(args):
    port_visits = port_visits_train_raw.loc[args['port_visits_row']]
    port_visits = port_visits[pd.notnull(port_visits['port_id'])]
    port = pd.get_dummies(port_visits['port_id'])
    feature = port.sum().max() / len(port_visits) if len(port_visits) > 0 else 0
    return pd.Series([feature],index=['main_port_weight'])


def average_meeting_duration(args):
    meetings = meetings_train_raw.loc[args['meeting_id1_row']]
    avg = PLACEHOLDER if meetings.empty else meetings['duration_min'].mean()
    return pd.Series([avg], index=['avg'])

def num_meetings(args):
    num = len(args['meeting_id1_row'])
    return pd.Series([num], index=['num'])

def meeting_in_country(args, country):
    meetings = meetings_train_raw.loc[args['meeting_id1_row']]
    return pd.Series([int(country in meetings['EEZ_name'].values)], index=['met in' + country])

def visit_in_country(args, country):
    meetings = port_visits_train_raw.loc[args['meeting_id1_row']]
    return pd.Series([int(country in meetings['country'].values)], index=['been in' + country])

def average_distance(ship_row):
    tic = time.time()
    meetings = meetings_train_raw[['port_id','start_time','duration_min','Lat','Long']].iloc[ship_row['meeting_id1_row']]
    meetings = meetings.dropna()
    if len(meetings)>0 :
        meetings['event_type'] = 'meeting'

    port_visits = port_visits_train_raw[['port_id','start_time','duration_min','Lat','Long']].iloc[ship_row['port_visits_row']]
    port_visits = port_visits.dropna()
    if len(port_visits)>0 :
        port_visits['event_type'] = 'port_visit'
    ship_vertices = meetings.append(port_visits);
    ship_vertices = ship_vertices.sort_values('start_time')
    start_times = ship_vertices['start_time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    vertex_durations = ship_vertices['duration_min'].apply(lambda x: timedelta(minutes=x))
    end_times = start_times + vertex_durations
    edge_durations = [start_times.iloc[i+1] - end_times.iloc[i] for i in range(0,len(start_times)-1)]
    edge_durations = [x.total_seconds() // 3600 for x in edge_durations]

    distances = np.zeros((len(ship_vertices)-1))
    speeds = np.zeros((len(ship_vertices)-1))

    for i in range(0,len(ship_vertices)-1):
        from_port = ship_vertices[['port_id']].iloc[[i]].values.tolist()[0][0]
        to_port =  ship_vertices[['port_id']].iloc[[i+1]].values.tolist()[0][0]
        edge_distance = int(great_circle(ship_vertices[['Lat','Long']].iloc[[i]].values.tolist()[0],ship_vertices[['Lat','Long']].iloc[[i+1]].values.tolist()[0]).kilometers)
        distances[i] = edge_distance
        speeds[i] = edge_distance / edge_durations[i] if edge_durations[i] > 0 else np.nan

    avg_dist = np.nanmean(distances)
    avg_speed = np.nanmean(speeds)
    std_dist = np.nanstd(distances)
    std_speed = np.nanstd(speeds)


    return pd.Series([avg_speed, std_speed])




#---------------------------------Learning Methods----------------------------------------

def classify_with_PCA(clf, train_set, train_labels, test_set, test_labels ):
    pca = PCA(n_components=50)
    pca.fit(train_set)
    train_set_new = pca.transform(train_set)
    test_set_new = pca.transform(test_set)
    return classify(clf, train_set_new, train_labels, test_set_new, test_labels)


def classify(clf, train_set, train_labels, test_set, test_labels):
    clf.fit(train_set, train_labels)
    new_labels =  clf.predict(test_set)
    error_vector = new_labels == test_labels
    accuracy = np.sum(error_vector) / len(error_vector)
    return accuracy, new_labels

def generate_feature_table(dataset, feature_funcs, normalize=False):
    full_table = None
    for func in feature_funcs:
        feature_table = dataset.apply(func, axis = 1)
        feature_table.fillna(value=0, inplace = True)

        if normalize:
            feature_range = feature_table.max() - feature_table.min();
            feature_table = (feature_table - feature_table.mean())/ feature_range
            feature_table.fillna(value=0, inplace=True)

        if full_table is None:
            full_table = feature_table
        else:
            full_table = pd.concat([full_table, feature_table], axis=1)

    if not np.all(np.isfinite(full_table)):
            pass

    return full_table

#----------------------------------The script---------------------------------------------
small_data = data_per_vessel[1:small_dataset_size]
country_funcs = []
visit_funcs = []

# for c in range(5):
#     country_func = lambda x : meeting_in_country(x, meeting_indicative_countries[c])
#     country_funcs.append(country_func)
#
# for v in range(5):
#     visit_func = lambda x : visit_in_country(x, port_indicative_countries[v])
#     visit_funcs.append(visit_func)

funcs = [port_id_probs, main_country_weight, average_distance]
feature_table = generate_feature_table(small_data, funcs, True)
train_set = feature_table[1:-test_set_size]
train_labels = small_data['type'].iloc[1:-test_set_size]


test_set = feature_table[-test_set_size:]
test_labels = small_data['type'].iloc[-test_set_size:]

#clf = svm.SVC()
clf = KNeighborsClassifier(n_neighbors=10)
if not np.all(np.isfinite(train_set)):
        pass
clf.fit(train_set, train_labels)
result  =  clf.predict(test_set)
error_vector = result == test_labels

accuracies = np.zeros((validation_n))

#cross-validation
chunk_size = (len(small_data) / validation_n)
for i in range(validation_n):
    test_range = range(i*validation_n,(i+1)*validation_n)
    new_test_set = feature_table.iloc[test_range]
    new_test_labels = small_data['type'].iloc[test_range]
    new_train_set = feature_table.drop(feature_table.index[test_range])
    new_train_labels = small_data['type'].drop(small_data.index[test_range])
    accuracy, new_labels = classify(clf, new_train_set, new_train_labels, new_test_set, new_test_labels)
    accuracies[i] = accuracy

print(accuracies)
print ('Accuracy:', np.mean(accuracies))






