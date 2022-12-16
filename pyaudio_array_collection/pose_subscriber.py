import numpy as np
from geometry_msgs.msg import PoseStamped
from rospy import Subscriber

class PoseSubscriber():

    def __init__(self, topic="/mocap_node/drone/pose"):
        self.pos = np.zeros(3)
        self.sub = Subscriber("pose_tracker", PoseStamped, callback=self.subscriber_callback)

    def subscriber_callback(self, data: PoseStamped):
        self.pos[0] = data.pose.x
        self.pos[1] = data.pose.y
        self.pos[2] = data.pose.z
    
    def get_pos(self):
        return self.pos
    