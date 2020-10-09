import os

edac_cmds = [
    'pysph run taylor_green --perturb 0.2 --openmp --scheme edac --nx 30 -d 00-0',
    'pysph run taylor_green --perturb 0.2 --openmp --scheme edac --nx 50 -d 00-1',
    'pysph run taylor_green --perturb 0.2 --openmp --scheme edac --nx 70 -d 00-2',
    'pysph run taylor_green --perturb 0.2 --openmp --scheme edac --nx 100 -d 00-3',
]

tvf_cmds = [
    'pysph run taylor_green --perturb 0.2 --openmp --scheme tvf --nx 30 -d 01-0',
    'pysph run taylor_green --perturb 0.2 --openmp --scheme tvf --nx 50 -d 01-1',
    'pysph run taylor_green --perturb 0.2 --openmp --scheme tvf --nx 70 -d 01-2',
    'pysph run taylor_green --perturb 0.2 --openmp --scheme tvf --nx 100 -d 01-3',
]

dpsph_cmds = [
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --nx 30 -d 02-0',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --nx 50 -d 02-1',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --nx 70 -d 02-2',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --nx 100 -d 02-3',
]

dpsph_LV_cmds = [
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --visc-correct --nx 30 -d 03-0',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --visc-correct --nx 50 -d 03-1',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --visc-correct --nx 70 -d 03-2',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --visc-correct --nx 100 -d 03-3',
]

dpsph_TVF_cmds = [
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --tvf-correct --nx 30 -d 04-0',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --tvf-correct --nx 50 -d 04-1',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --tvf-correct --nx 70 -d 04-2',
    'python Taylor_Green-PySPH.py --perturb 0.2 --openmp --tvf-correct --nx 100 -d 04-3',
]


myCmds = [edac_cmds, tvf_cmds, dpsph_cmds, dpsph_LV_cmds, dpsph_TVF_cmds]

for group in myCmds:
    for doCmd in group:
        os.system(doCmd)
        os.system('cls')