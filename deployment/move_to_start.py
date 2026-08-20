# Move the robot to the predefined initial configuration
import logging

import panda_py

logging.basicConfig(level=logging.INFO)

panda = panda_py.Panda('192.168.1.8')
panda.move_to_start()

print("done")
