# Test: gripper control (grasp and open)
import logging

from panda_py import libfranka

logging.basicConfig(level=logging.INFO)

gripper = libfranka.Gripper('192.168.1.8')

gripper.grasp(0, 0.2, 10, 0.04, 0.04)
gripper.move(0.08, 0.2)

print("done")
