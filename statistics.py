import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sequences_length_stat(data=None):
    if data is not None:
        sequences = data.groupby(['SequenceID'])
        sequences = sequences.groups
        hist={}
        for key in sequences:
            seq_len = len(sequences[key])
            if hist.has_key(seq_len):
                hist[seq_len]+=1
            else:
                hist[seq_len]=1
        #len(df.SequenceID.unique())==sum(hist.values())  # validate
    else:
        res ={1: 25342, 2: 21971, 3: 15963, 4: 13815, 5: 11391, 6: 9814, 7: 8689, 8: 7620, 9: 6661, 10: 6170, 11: 5315, 12: 4872, 13: 4397, 14: 3896, 15: 3552, 16: 3170, 17: 2981, 18: 2645, 19: 2435, 20: 2180, 21: 1971, 22: 1823, 23: 1736, 24: 1630, 25: 1511, 26: 1346, 27: 1205, 28: 1134, 29: 1087, 30: 962, 31: 864, 32: 853, 33: 740, 34: 740, 35: 652, 36: 618, 37: 594, 38: 494, 39: 456, 40: 449, 41: 444, 42: 392, 43: 355, 44: 378, 45: 327, 46: 309, 47: 268, 48: 261, 49: 269, 50: 244, 51: 195, 52: 215, 53: 233, 54: 210, 55: 177, 56: 181, 57: 165, 58: 151, 59: 136, 60: 134, 61: 168, 62: 123, 63: 103, 64: 100, 65: 98, 66: 106, 67: 84, 68: 92, 69: 90, 70: 82, 71: 71, 72: 66, 73: 71, 74: 66, 75: 71, 76: 62, 77: 61, 78: 60, 79: 58, 80: 47, 81: 44, 82: 64, 83: 48, 84: 37, 85: 38, 86: 40, 87: 41, 88: 33, 89: 34, 90: 40, 91: 33, 92: 31, 93: 34, 94: 34, 95: 31, 96: 31, 97: 27, 98: 39, 99: 23, 100: 17, 101: 30, 102: 26, 103: 23, 104: 27, 105: 18, 106: 27, 107: 21, 108: 18, 109: 16, 110: 20, 111: 23, 112: 16, 113: 14, 114: 18, 115: 7, 116: 15, 117: 21, 118: 17, 119: 16, 120: 17, 121: 8, 122: 17, 123: 13, 124: 13, 125: 13, 126: 10, 127: 13, 128: 13, 129: 10, 130: 11, 131: 10, 132: 12, 133: 6, 134: 7, 135: 9, 136: 10, 137: 12, 138: 7, 139: 12, 140: 11, 141: 8, 142: 15, 143: 3, 144: 6, 145: 7, 146: 3, 147: 9, 148: 6, 149: 4, 150: 10, 151: 6, 152: 9, 153: 6, 154: 7, 155: 8, 156: 6, 157: 5, 158: 7, 159: 7, 160: 6, 161: 7, 162: 5, 619: 1, 164: 4, 165: 1, 166: 3, 167: 3, 168: 4, 169: 4, 170: 3, 171: 3, 172: 3, 173: 8, 174: 3, 175: 8, 176: 2, 177: 2, 178: 5, 179: 3, 180: 7, 181: 4, 182: 1, 183: 3, 184: 1, 185: 4, 187: 4, 188: 4, 189: 2, 190: 2, 191: 2, 192: 3, 193: 3, 194: 2, 195: 3, 196: 3, 197: 5, 198: 3, 545: 1, 200: 3, 201: 1, 202: 4, 203: 3, 205: 2, 206: 1, 207: 1, 208: 3, 209: 1, 210: 4, 211: 2, 212: 2, 213: 6, 214: 1, 215: 4, 216: 2, 217: 1, 218: 2, 219: 3, 220: 3, 222: 2, 224: 2, 635: 1, 228: 2, 229: 2, 230: 1, 231: 2, 232: 2, 235: 2, 236: 2, 237: 2, 238: 1, 239: 1, 241: 1, 242: 1, 245: 1, 248: 1, 250: 1, 251: 1, 252: 1, 253: 1, 254: 1, 256: 1, 257: 1, 258: 1, 618: 1, 260: 2, 261: 3, 263: 3, 264: 1, 265: 1, 268: 1, 272: 1, 279: 1, 281: 3, 199: 1, 283: 1, 285: 1, 287: 1, 534: 2, 292: 1, 293: 1, 294: 2, 298: 1, 299: 1, 300: 1, 302: 1, 303: 2, 163: 2, 307: 1, 309: 1, 313: 1, 607: 1, 319: 1, 320: 1, 522: 1, 323: 1, 325: 2, 334: 1, 335: 1, 338: 1, 339: 1, 340: 1, 343: 1, 346: 1, 348: 1, 540: 1, 352: 1, 354: 1, 571: 1, 633: 1, 358: 1, 361: 1, 363: 1, 370: 1, 371: 2, 374: 2, 375: 1, 380: 1, 389: 1, 2144: 1, 559: 1, 410: 1, 413: 1, 416: 1, 428: 1, 754: 1, 304: 1, 434: 1, 436: 1, 44470: 1, 442: 1, 444: 1, 467: 1, 482: 1, 491: 1, 595: 1}
        plt.bar(range(1,100), res.values()[:99], align='center')
        plt.xticks(range(1,100), res.keys()[:99])
        plt.show()

def platform_usage(channel_map,data=None):
    if data is not None:
        platforms = data.groupby(['ChannelTypeID'])
        platforms = platforms.groups
        plat_hist={}
        for key in platforms:
            plat_hist[key] = len(platforms[key])
        channel_map = pd.read_csv(channel_map,index_col=0)
        channel_map = channel_map.to_dict()[' Unknown']
        new_hist = {}
        for plat in plat_hist:
            new_hist[channel_map[plat]]=plat_hist[plat]
    else:
        res2 = {2: 343635, 5: 230526, 14: 1626914, 17: 2668, 18: 1245, 20: 2633, 21: 5441, 22: 129, 23: 198, 24: 32148, 25: 9988, 26: 1888, 27: 13268, 28: 609, 29: 292}
        res={' Web': 230526, ' RetailAffiliatePartners': 129, ' UnChurnCustomerTriggered': 609, ' RetailIndirectNational': 1888, ' RetailIndirectLocal': 9988, ' ChurnCustomerTriggered': 5441, ' RetailTelesales': 13268, ' IVR': 1626914, ' RetailCompanyOwnedAndKiosks': 32148, ' UnChurnSystemTriggered': 292, ' Voice': 343635, ' ChurnSystemTriggered': 2633, ' Warehouse': 2668, ' RetailBusinessSales': 198, ' Insurance': 1245}
        plt.bar(range(len(res)), res.values(), align='center')
        plt.xticks(range(len(res)), res2.keys())
        plt.show()


def channel_movements(channel_map,data=None):
    if data is not None:
        transition_matrix = np.zeros([data.channels['0'].max()+1,data.channels['0'].max()+1])
        prob_matrix = np.zeros([data.channels['0'].max()+1,data.channels['0'].max()+1])
        curr_channel = data.df.ChannelTypeID.values
        prev_channel = data.df.PreviousChannelTypeID.values
        assert(len(curr_channel==len(prev_channel)))
        for i in range(len(curr_channel)):
            transition_matrix[prev_channel[i]][curr_channel[i]]+=1
        for i in range(transition_matrix.shape[0]):
            prob_matrix[i,:] = transition_matrix[i,:]/transition_matrix[i,:].sum()
        channel_map = pd.read_csv(channel_map,index_col=0)
        channel_map = channel_map.to_dict()[' Unknown']
        channel_map[0]='Unknown'
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[0]):
                if prob_matrix[i,j]>0.01:
                    print channel_map[i] + '  -->  ' + channel_map[j] +'  prob: ' + str(prob_matrix[i,j])

    else:
        res2 = {2: 343635, 5: 230526, 14: 1626914, 17: 2668, 18: 1245, 20: 2633, 21: 5441, 22: 129, 23: 198, 24: 32148,
                25: 9988, 26: 1888, 27: 13268, 28: 609, 29: 292}
        # res = {' Web': 230526, ' RetailAffiliatePartners': 129, ' UnChurnCustomerTriggered': 609,
        #        ' RetailIndirectNational': 1888, ' RetailIndirectLocal': 9988, ' ChurnCustomerTriggered': 5441,
        #        ' RetailTelesales': 13268, ' IVR': 1626914, ' RetailCompanyOwnedAndKiosks': 32148,
        #        ' UnChurnSystemTriggered': 292, ' Voice': 343635, ' ChurnSystemTriggered': 2633, ' Warehouse': 2668,
        #        ' RetailBusinessSales': 198, ' Insurance': 1245}
        # plt.bar(range(len(res)), res.values(), align='center')
        # plt.xticks(range(len(res)), res2.keys())
        # plt.show()
