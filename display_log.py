import numpy as np

raw = np.load("experiment_log.npy", allow_pickle=True)
data = raw.item() if raw.shape == () else raw

O_T_EE = data['O_T_EE']  # list of 700 个 (16,) array
time = data['time']

# 关键：列主序 reshape，取平移部分
def extract_position(flat_16):
    T = np.array(flat_16).reshape(4, 4, order='F')  # order='F' = 列主序
    return T[:3, 3]

positions = np.array([extract_position(p) for p in O_T_EE])

start_pos = positions[0]
final_pos = positions[-1]

print("起始位置:", start_pos)
print("最终位置:", final_pos)
print("总步数:", len(positions))
print("总时长(ms):", time[-1][0] - time[0][0])