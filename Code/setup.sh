'''
YUEXIAOLI---CATHI
'''

#!/usr/bin/env bash

# create the directories to store results locally
save_dir='/tf_data_traj/'
sudo mkdir -p $save_dir'/data/'
sudo mkdir -p $save_dir'/nn_models/'
sudo mkdir -p $save_dir'/results/'
sudo chown -R "$USER" $save_dir

# copy train and test data with proper naming
data_dir='/CATHI/data'
cp $data_dir'/data_train.txt' $save_dir'/data/chat.in'
cp $data_dir'/data_combination.txt' $save_dir'/data/chatvoc.in'
cp $data_dir'/data_test.txt' $save_dir'/data/chat_test.in'