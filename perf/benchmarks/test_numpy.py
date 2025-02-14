import time
import numpy as np


names = ['1e+0', '1e+1', '1e+2', '1e+3']
sizes = [1, 10, 100, 1000]

cnt = 0
max = np.power(len(sizes), 3)
for n1, s1 in zip(names, sizes):
    for n2, s2 in zip(names, sizes):
        for n3, s3 in zip(names, sizes):
            if all([s >= 1000 for s in [s1, s2, s3]]): continue

            a = np.random.rand(s1, s2, s3)
            b = np.random.rand(s1, s2, s3)

            cnt += 1
            for i in range(10):
                print(f'{cnt}/{max} - {i}')

                try:
                    start_time = time.time()
                    res = np.add(a, b)
                    end_time = time.time()
                    print('>>> {:10} {:6} {:6} {:6} @ {:.6f}'.format('add', n1, n2, n3, round(np.subtract(end_time, start_time), 6)))
                except: pass

                try:
                    start_time = time.time()
                    res = np.subtract(a, b)
                    end_time = time.time()
                    print('>>> {:10} {:6} {:6} {:6} @ {:.6f}'.format('subtract', n1, n2, n3, round(np.subtract(end_time, start_time), 6)))
                except: pass

                try:
                    start_time = time.time()
                    res = np.multiply(a, b)
                    end_time = time.time()
                    print('>>> {:10} {:6} {:6} {:6} @ {:.6f}'.format('multiply', n1, n2, n3, round(np.subtract(end_time, start_time), 6)))
                except: pass

                try:
                    start_time = time.time()
                    res = np.divide(a, b)
                    end_time = time.time()
                    print('>>> {:10} {:6} {:6} {:6} @ {:.6f}'.format('divide', n1, n2, n3, round(np.subtract(end_time, start_time), 6)))
                except: pass

                try:
                    start_time = time.time()
                    res = np.linalg.norm(a)
                    res = np.linalg.norm(b)
                    end_time = time.time()
                    print('>>> {:10} {:6} {:6} {:6} @ {:.6f}'.format('norm', n1, n2, n3, round(np.subtract(end_time, start_time), 6)))
                except: pass

                if all([s < 1000 for s in [s1, s2, s3]]):
                   try:
                       start_time = time.time()
                       res = np.dot(a, b)
                       end_time = time.time()
                       print('>>> {:10} {:6} {:6} {:6} @ {:.6f}'.format('dot', n1, n2, n3, round(np.subtract(end_time, start_time), 6)))
                   except: pass
