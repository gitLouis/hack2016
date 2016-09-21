import numpy
import pandas

d = {'one' : pandas.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two' : pandas.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pandas.DataFrame(d)

df1 = pandas.read_csv("C:\\Users\\S Louis\\Dropbox\\Sea Snails\\sample data\\meetings_train_sample.csv")
df2 = pandas.read_csv("C:\\Users\\S Louis\\Dropbox\\Sea Snails\\raw data\\port_visits_train.csv")

    #change
print(df1)