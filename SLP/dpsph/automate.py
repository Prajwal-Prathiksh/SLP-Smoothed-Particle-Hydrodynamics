import os

eul_cmds = [
    'python test_integrator.py --INT eul --N 40 -d 00',
    'python test_integrator.py --INT eul --N 80 -d 01',
    'python test_integrator.py --INT eul --N 160 -d 02',
    'python test_integrator.py --INT eul --N 320 -d 03',
    'python test_integrator.py --INT eul --N 640 -d 04',
]

rk4_cmds = [
    'python test_integrator.py --INT rk4 --N 40 -d 10',
    'python test_integrator.py --INT rk4 --N 80 -d 11',
    'python test_integrator.py --INT rk4 --N 160 -d 12',
    'python test_integrator.py --INT rk4 --N 320 -d 13',
    'python test_integrator.py --INT rk4 --N 640 -d 14',
]


myCmds =  [eul_cmds,]

for group in myCmds:
    for doCmd in group:
        os.system('cls')
        os.system(doCmd)