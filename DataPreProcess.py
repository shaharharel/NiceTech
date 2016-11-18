import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import json
import re
#import simplejson

def default(o):
    if isinstance(o, np.integer): return int(o)
    raise TypeError

class NiceData():
    def __init__(self,data_path,feature_mapping,channel_mapping):
        self.channels = pd.read_csv(channel_mapping)
        self.feature_map = pd.read_csv(feature_mapping)
        self.data = self.create_data_frame(data_path)
        self.channel_map  = channel_mapping
        self.adjust_data()

    def create_data_frame(self,data_path):
        col = self.feature_map.name
        self.df = pd.read_csv('Magnet.csv',delimiter='|',names=col,usecols=[i for i in range(50) if i not in [2,7,21,22,23,24,25,26,27,29,39]])

    def adjust_data(self):
        # add dummies
        for i in self.df.columns.values:
            if i[0:16]=='DynamicAttribute' or i not in ['NextChannelTypeID','SessionID','SequenceID','PlaceInSequence','LastSessionInSequence','DateID','SessionStartTimeUTC','SessionEndTimeUTC','SessionDuration','PlatformCustomerID','FirstDateID','Master_BAN','TimeInterval']:
                self.df = pd.concat([self.df, pd.get_dummies(self.df[i],prefix=[i])], axis=1)
                self.df = self.df.drop([i], axis=1)
        for i in ['SessionStartTimeUTC','SessionEndTimeUTC','DateID','FirstDateID']:
            temp = pd.to_datetime(self.df[i])
            temp = (temp - temp.min()) / np.timedelta64(1, 'D')
            self.df = pd.concat([self.df, temp], axis=1)
            self.df = self.df.drop([i], axis=1)

    def create_sequences(self):
        sequences = self.df.groupby(['SequenceID'])
        sequences = sequences.groups
        train_data={}
        target={}
        for p, key in enumerate(sequences):
            print "sequence " + str(p)
            if p<69369:
                continue
            if p==120000:
                break
            data_point = None
            seq_target=[]
            sorted_indexs = self.df.ix[sequences[key].values].sort('PlaceInSequence').index.values
            for idx in sorted_indexs:
                seq_target.append(self.df.ix[idx].NextChannelTypeID)
                temp = self.df.ix[idx].drop(['Master_BAN','SessionID','SequenceID','PlatformCustomerID','PlaceInSequence','NextChannelTypeID','LastSessionInSequence'])
                if data_point is None:
                    data_point = temp.as_matrix()
                else:
                    data_point = np.concatenate((data_point,temp))
            #data.df = data.df.drop(sorted_indexs)
            train_data[key]=list(data_point)
            target[key] = seq_target
        self.write_to_json('new_data3.json',train_data)
        self.write_to_json('channel_target3.json',target)

    def append_to_json(self,path,key,value):
        with open(path) as json_file:
            json_decoded = json.load(json_file)
        json_decoded[key] = value
        with open(path, 'w') as json_file:
            json.dump(json_decoded, json_file,default=default)

    def write_to_json(self,path,data):
        with open(path, 'w') as outfile:
            json.dump(data, outfile,default=default)

    def statistics(self):
        statistics.sequences_length_stat(self.df)
        statistics.platform_usage(self.channels, self.df)
        statistics.channel_movements(self.channels, self.df)

if __name__ == '__main__':
    #statistics.platform_usage('DataDesc.csv')
    #statistics.sequences_length_stat()
    data = NiceData('Magnet.csv','DataDesc.csv','DataDesc2.csv')
    data.create_sequences()
