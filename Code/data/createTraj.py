'''
YUEXIAOLI---CATHI
'''

import os, sys, time, pickle, tempfile
import math, random, itertools
import pandas as pd


fpoi = os.path.join('/CATHI/code/data/poi-Magick.csv')
poi_all = pd.read_csv(fpoi)
poi_all.set_index('poiID', inplace=True)

ftraj = os.path.join('/CATHI/code/data/traj-Magick.csv')
traj_all = pd.read_csv(ftraj)

def extract_traj(tid, traj_all):
    traj = traj_all[traj_all['trajID'] == tid].copy()
    traj.sort_values(by=['startTime'], ascending=True, inplace=True)
    return traj['poiID'].tolist()

trajid_set_all = sorted(traj_all['trajID'].unique().tolist())
traj_dict = dict()
for trajid in trajid_set_all:
    traj = extract_traj(trajid, traj_all)
    assert(trajid not in traj_dict)
    traj_dict[trajid] = traj

savefile = open('/CATHI/code/data/magickdata.txt','w')
for i in traj_dict:
    j=traj_dict[i]
    if len(j)>2:
        for x in j:
            savefile.write(str(x)+' ')
        savefile.write('\n')