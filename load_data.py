
import pandas
from os.path import expanduser
try:
   import cPickle as pickle
except:
   import pickle


home = expanduser("~")
ports_probs = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\feature data\\ship_type_by_port_id_prob.csv",  index_col='port_id')
port_country_probs = pandas.read_csv(home+"\\Dropbox\\Sea Snails\\feature data\\ship_type_by_port_country.csv", index_col='port_country')

pickle.dump(ports_probs, open( home+"\\Dropbox\\Sea Snails\\feature data\\ship_type_by_port_id_prob.p", "wb" ) )
pickle.dump(port_country_probs, open( home+"\\Dropbox\\Sea Snails\\feature data\\ship_type_by_port_country.p", "wb" ) )