from geometry_msgs.msg import PoseStamped
from rospy import Subscriber

class PoseSubscriber():

    def __init__(self, topic="/mocap_node/drone/pose"):
        self.x = None
        self.y = None
        self.z = None
        self.sub = Subscriber("pose_tracker", PoseStamped, callback=self.subscriber_callback)

    def subscriber_callback(self, data: PoseStamped):
        self.x = data.pose.x
        self.y = data.pose.y
        self.z = data.pose.z
    