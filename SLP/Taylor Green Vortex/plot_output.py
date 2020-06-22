import numpy as np
from matplotlib import pyplot as plt

nx, perturb = 50, 0
nx, perturb = str(int(nx)), str(int(perturb))

file_base = '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics/SLP/Taylor Green Vortex/Output/'
file_base += 'nx_' + nx + '/perturb_' + perturb

sph_schm = ['edac', 'tvf', 'wcsph', 'dpsph']

################################################################################
## Plot decay
################################################################################
sz = (15,10)
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, decay, decay_ex = data['t'], data['decay'], data['decay_ex']
    
    # Plot
    if schm == 'edac':
        plt.semilogy(t, decay_ex, '--k', linewidth=3, label="exact")
    plt.semilogy(t, decay, linewidth=2, label=schm)
    
# Formatting    
plt.xlabel(r't $\rightarrow$', fontsize=15)
plt.ylabel(r'Max Velocity $\rightarrow$',fontsize=15)
tle = 'Max Velocity in flow vs Time | nx = ' + nx + ' | perturb = ' + perturb + ' |'
plt.title(tle,fontsize=20)
plt.legend()
tle = 'decay_' + nx + '.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot L_{\infty} error
################################################################################
sz = (15,10)
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, linf = data['t'], data['linf']
    
    # Plot
    plt.plot(t, linf, linewidth=2, label=schm)
    
# Formatting
plt.xlabel(r't $\rightarrow$',fontsize=15)
plt.ylabel(r'$L_\infty$ error $\rightarrow$',fontsize=15)
tle = r'$L_\infty$ error vs Time | nx = ' + nx + ' | perturb = ' + perturb + ' |'
plt.title(tle,fontsize=20)
plt.legend()
tle = 'linf_error_' + nx + '.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot L_1 error
################################################################################
sz = (15,10)
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, l1 = data['t'], data['l1']
    
    # Plot
    plt.plot(t, l1,linewidth=2, label=schm)
    
# Formatting
plt.xlabel(r't $\rightarrow$',fontsize=15)
plt.ylabel(r'$L_1$ error $\rightarrow$',fontsize=15)
tle = r'$L_1$ error vs Time | nx = ' + nx + ' | perturb = ' + perturb + ' |'
plt.title(tle,fontsize=20)
plt.legend()
tle = 'l1_error_' + nx + '.png' 
plt.savefig(tle, dpi=400)

################################################################################
## Plot L_1 error for p
################################################################################
sz = (15,10)
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, p_l1 = data['t'], data['p_l1']
    
    # Plot
    plt.plot(t, p_l1,linewidth=2, label=schm)
    
# Formatting
plt.xlabel('t',fontsize=15)
plt.ylabel(r'$L_1$ error for $p$',fontsize=15)
tle = r'$L_1$ error for $p$ vs Time | nx = ' + nx + ' | perturb = ' + perturb + ' |'
plt.title(tle,fontsize=20)
plt.legend()
tle = 'p_l1_error_' + nx + '.png' 
plt.savefig(tle, dpi=400)



