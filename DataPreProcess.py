import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
class NiceData():
    def __init__(self,data_path,feature_mapping,channel_mapping):
        self.channels = pd.read_csv(channel_mapping)
        self.feature_map = pd.read_csv(feature_mapping)
        self.data = self.create_data_frame(data_path)
        self.channel_map  = channel_mapping
    def create_data_frame(self,data_path):
        col = self.feature_map.name
        self.df = pd.read_csv('Magnet.csv',delimiter='|',names=col,usecols=[i for i in range(50)])
    def statistics(self):
        statistics.sequences_length_stat(self.df)
        statistics.platform_usage(self.df,self.channels_map)
    def validate_sequence_order(self,data):
        return
if __name__ == '__main__':
    #statistics.platform_usage('DataDesc.csv')
    statistics.sequences_length_stat()
    #data = NiceData('Magnet.csv','DataDesc.csv','DataDesc2.csv')