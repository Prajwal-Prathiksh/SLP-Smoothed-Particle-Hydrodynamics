import numpy as np
import matplotlib.pyplot as plt
from pysph.tools.pprocess import get_ke_history
from pysph.tools.interpolator import Interpolator
from pysph.solver.utils import load
from pysph.examples.ghia_cavity_data import get_u_vs_y, get_v_vs_x
plt.rcParams.update({'font.size': 20})

file_base = '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics/SLP/Cavity/PySPH-Testing'

sph_schm = ['00', '01']

sph_schm_legend = {
    '00': 'PySPH: TVF', '01': 'Euler Int'
}
title_additional = ' | nx = 50'
savefig_additional = ''

################################################################################
## Plot KE-History
################################################################################
sz = (14.40,10.80)
plt.figure(figsize=sz)

for schm in sph_schm:        
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, ke = data['t'], data['ke']
    
    # Plot
    plt.plot(t, ke, linewidth=2, label=sph_schm_legend[schm])
    
# Formatting    
plt.xlabel('t')
plt.ylabel('Kinetic energy')
tle = r'Kinetic Energy History' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/ke_history' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot Centerline
################################################################################
_x = np.linspace(0, 1, 101)
xx, yy = np.meshgrid(_x, _x)
re = 100
plt.figure(figsize=sz)
s1 = plt.subplot(211)
s2 = plt.subplot(212)

cnt = 0
for schm in sph_schm:
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)

    ui, vi, t = data['u'], data['v'], data['t']
    vmag = np.sqrt(ui**2 + vi**2)
    tf = t[-1]
    ui_c = ui[:, 50]
    vi_c = vi[50]

    
    
    if cnt == 0:
        y, data = get_u_vs_y()
        if re in data:
            s1.plot(data[re], y, 'o', fillstyle='none', label='Ghia et al.')
    s1.plot(ui_c, _x, label=sph_schm_legend[schm])

    if cnt == 0:
        x, data = get_v_vs_x()
        if re in data:
            s2.plot(x, data[re], 'o', fillstyle='none', label='Ghia et al.')
        cnt = 1    
    s2.plot(_x, vi_c, label=sph_schm_legend[schm])


s1.set_xlabel(r'$v_x$')
s1.set_ylabel(r'$y$')
s1.legend()
s2.set_xlabel(r'$x$')
s2.set_ylabel(r'$v_y$')
s2.legend()
tle = file_base + '/centerline' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)

################################################################################
## Plot Streamplot
################################################################################
for schm in sph_schm:
    plt.figure(figsize=sz)
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)

    ui, vi, t = data['u'], data['v'], data['t']
    vmag = np.sqrt(ui**2 + vi**2)
    tf = t[-1]

    plt.streamplot(
        xx, yy, ui, vi, density=(2, 2),  # linewidth=5*vmag/vmag.max(),
        color=vmag
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Streamlines at %s seconds' % tf)
    tle = file_base + '/streamplot_' + schm + '.png' 
    plt.savefig(tle, dpi=400)