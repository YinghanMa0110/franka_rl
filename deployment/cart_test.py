# import logging 
# import panda_py
# from panda_py import libfranka

# panda_py.Panda('192.168.1.8')
# gripper = libfranka.Gripper('192.168.1.8')
# logging.basicConfig(level=logging.INFO)


# panda.move_to_start()
# pose = panda.get_pose()
# pose[2,3] -= .1
# q = panda_py.ik(pose)
# panda.move_to_joint_position(q)
import panda_py
from panda_py import controllers

robot = panda_py.Panda('192.168.1.8')

robot.move_to_start()

ctrl = controllers.CartesianImpedance(
    filter_coeff=1.0
)

robot.start_controller(ctrl)

pos = robot.get_position()
q = robot.get_orientation()

target = pos.copy()
target[2] += 0.1# z方向2cm

with robot.create_context(
    frequency=1000,
    max_runtime=10
) as ctx:

    while ctx.ok():
        ctrl.set_control(target, q)

print("done")
