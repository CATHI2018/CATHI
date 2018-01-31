import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from lib.Trajectory_Prediction import Trajectory_Prediction


def main(_):
    Trajectory_Prediction()

if __name__ == "__main__":
    tf.app.run()
