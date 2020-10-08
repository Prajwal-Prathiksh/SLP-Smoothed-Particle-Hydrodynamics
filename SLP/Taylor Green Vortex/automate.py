import os

myCmds = [
    'pysph run taylor_green --perturb 0.2 --openmp --nx 30 --scheme edac -d 01-0',
    'pysph run taylor_green --perturb 0.2 --openmp --nx 50 --scheme edac -d 01-1',
    'pysph run taylor_green --perturb 0.2 --openmp --nx 100 --scheme edac -d 01-2',
    'pysph run taylor_green --perturb 0.2 --openmp --nx 150 --scheme edac -d 01-3',
]

for doCmd in myCmds:
    os.system(doCmd)
    os.system('cls')