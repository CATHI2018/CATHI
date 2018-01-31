import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from lib.train import train


def main(_):
    train()

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    try:
        tf.app.run()
    except KeyboardInterrupt:
        endtime = datetime.datetime.now()
        interval=(endtime - starttime).seconds
        logfile = open('timelog.txt' ,'a+')
        s = 'starttime:'+str(starttime)+'\t'+'endtime:'+str(endtime)+'\t'+' used '+str(interval)+'seconds'+'\n'
        logfile.write(s)
