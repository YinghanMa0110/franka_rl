# Test: read robot state (joint positions/velocities/torques, end-effector pose)
import logging

import numpy as np
import panda_py

logging.basicConfig(level=logging.INFO)

panda = panda_py.Panda('192.168.1.8')
state = panda.get_state()

print("Real Franka state")
print(f"q(joint positions):shape=({len(state.q)},),value={np.array(state.q).round(3)}")
print(f"dq(joint velocities):shape=({len(state.dq)},),value={np.array(state.dq).round(3)}")
print(f"tau_J(joint torques):shape=({len(state.tau_J)},),value={np.array(state.tau_J).round(3)}")
print(f"O_T_EE(ee pose):shape=({len(state.O_T_EE)},)")

# O_T_EE is column-major, so reshape with order='F'
ee_matrix = np.array(state.O_T_EE).reshape(4, 4, order='F')
ee_pos = ee_matrix[:3, 3]
print(f"End-effector xyz:{ee_pos.round(3)}")
