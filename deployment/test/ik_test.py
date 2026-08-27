# Test: inverse kinematics motion (Cartesian pose -> IK -> joint position command)
import logging

import panda_py

logging.basicConfig(level=logging.INFO)

panda = panda_py.Panda('192.168.1.8')
panda.move_to_start()

pose = panda.get_pose()
pose[2, 3] -= .1  # move 10cm down in z
q = panda_py.ik(pose)
panda.move_to_joint_position(q)

print("done")
