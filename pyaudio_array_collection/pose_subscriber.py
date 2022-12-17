import numpy as np
from geometry_msgs.msg import PoseStamped
from rospy import Subscriber, init_node

class PoseSubscriber():

    def __init__(self, topic="/mocap_node/drone/pose"):
        self.pos = np.zeros(3)
        init_node("pose_subscriber")
        self.sub = Subscriber(topic, PoseStamped, callback=self.subscriber_callback, queue_size=10)

    def subscriber_callback(self, data: PoseStamped):
        self.pos[0] = data.pose.position.x
        self.pos[1] = data.pose.position.y
        self.pos[2] = data.pose.position.z
    
    def get_pos(self):
        return self.pos.copy()
    
if __name__ == "__main__":
    sub = PoseSubscriber()
    from time import sleep
    while True:
        sleep(0.1)
        print(sub.pos)