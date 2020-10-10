import os

dpsph_cmds = [
    #'python Cavity-PySPH.py --openmp --pfreq 1000 --re 1000 -d 03-1',
    'python Cavity-PySPH.py --openmp --pfreq 2000 --re-correct --re 10000 -d 03-2',
    'python Cavity-PySPH.py --openmp --pfreq 2000 --re-correct --re 10000 -d 03-3 --kernel CubicSpline',
]


myCmds = [dpsph_cmds]

for group in myCmds:
    for doCmd in group:
        os.system('cls')
        os.system(doCmd)