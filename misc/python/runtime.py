import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta


labels = {
    'eval_0 t1': ( 10000,  10000),
    'eval_0 t2': ( 10000,  10000),
    'eval_0 t3': ( 10000,  10000),
    'train t1':  (100000, 100000),
    'eval_1 t1': ( 10000,  10000),
    'eval_1 t2': ( 10000,  10000),
    'eval_1 t3': ( 10000,  10000),
    'train t2':  (100000, 100000),
    'eval_2 t1': ( 10000,  10000),
    'eval_2 t2': ( 10000,  10000),
    'eval_2 t3': ( 10000,  10000),
    'train t3':  ( 86072, 100000),
    'eval_3 t1': (     0,  10000),
    'eval_3 t2': (     0,  10000),
    'eval_3 t3': (     0,  10000),
}

raw = '''\
2023-10-21 00:11:15,614 - file_logger - WARNING - Switch training False on track straight
2023-10-21 00:29:34,745 - file_logger - WARNING - Switch training False on track jojo_r
2023-10-21 00:43:00,071 - file_logger - WARNING - Switch training False on track slalom_sym
2023-10-21 00:55:55,421 - file_logger - WARNING - Switch training True on track straight
2023-10-21 05:10:09,535 - file_logger - WARNING - Switch training False on track straight
2023-10-21 05:25:42,230 - file_logger - WARNING - Switch training False on track jojo_r
2023-10-21 05:37:02,049 - file_logger - WARNING - Switch training False on track slalom_sym
2023-10-21 05:55:40,767 - file_logger - WARNING - Switch training True on track snake
2023-10-22 18:04:08,389 - file_logger - WARNING - Switch training False on track straight
2023-10-22 18:14:36,858 - file_logger - WARNING - Switch training False on track jojo_r
2023-10-22 18:25:08,455 - file_logger - WARNING - Switch training False on track slalom_sym
2023-10-22 18:35:34,730 - file_logger - WARNING - Switch training True on track puzzle
'''

timestamps = [datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f') for line in raw.splitlines()]
timestamps.append(datetime.now())

elapsed = np.array([x.total_seconds() for x in np.diff(timestamps)])
elapsed = np.concatenate([elapsed, np.array([
    np.average(elapsed[0::4]),
    np.average(elapsed[1::4]),
    np.average(elapsed[2::4]),
])], axis=0)

keys = np.array(list(labels.keys()))
values = np.array(list(labels.values()))

fig, axs = plt.subplots(2, 1)

axs[0].set_title('seconds per timestep')
tmp1 = np.divide(elapsed, values[:, 0])
tmp1[values[:, 0] == 0] = 0
axs[0].bar(keys, tmp1)
tmp2 = np.divide(elapsed, values[:, 1])
tmp2[values[:, 0] != 0] = 0
axs[0].bar(keys, tmp2)
axs[0].legend(['elapsed', 'remaining'])

axs[1].set_title('duration in minutes')
tmp3 = np.divide(elapsed, 60)
tmp3[values[:, 0] == 0] = 0
axs[1].bar(keys, tmp3)
tmp4 = np.divide(np.multiply(np.add(tmp1, tmp2), np.subtract(values[:, 1], values[:, 0])), 60)
# tmp4[values[:, 0] != 0] = 0
axs[1].bar(keys, tmp4, bottom=tmp3)
axs[1].legend(['elapsed', 'remaining'])

total_elapsed = np.sum(tmp3)
total_remaining = np.sum(tmp4)

print('total_elapsed (in h):', np.divide(total_elapsed, 60))
print('total_remaining (in h):', np.divide(total_remaining, 60))

print('eta:', datetime.now() + timedelta(seconds=total_remaining * 60))

plt.show()
