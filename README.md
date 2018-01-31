# Datasets

There are three datasets: Flicker,Geolife and Tokyo.

##The format of dataset, for example:
poi-Edin.csv POI data in Edinburgh.
      * poiID: POI identity
      * poiCat: POI category
      * poiLon: POI longitude
      * poiLat: POI latitude
traj-Edin.csv Trajectories in Edinburgh.
      * userID: User identity
      * trajID: Trajectory identity
      * poiID: POI identity
      * startTime: Timestamp that the user started to visit this POI
      * endTime: Timestamp that the user left this POI
      * #photo: Number of photos taken by the user at this POI
      * trajLen: Number of POIs visited in this trajectory by the user
      * poiDuration: The visit duration (seconds) at this POI by the user

You can click https://sites.google.com/site/limkwanhui/datacode#ijcai15 to download the original dataset of Flickr.

# Usage
To generate the results from scratch, please follow these five steps:

* Creat the trajectory by excute /CATHI/code/data/createTraj.py.
* Use `bash setup.sh` to create the directories to store results locally, and copy train and test data with proper naming.
* Run `train.py` to train the model. You will get the ckpt file at path: /code/tf_data_traj/nn_models.
* Sequentially run `Trajectory_Prediction.py` to predict trajectory. The results are in /code/tf_data_traj/results.
* Finally, you can calculate F1 and pairs-F1 by running `F1AndPairsF1.py`.

# Requirements

python:2.7.12
tensorflow-gpu : 1.0.0