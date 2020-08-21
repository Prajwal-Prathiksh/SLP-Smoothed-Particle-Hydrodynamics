import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


nx, perturb = 50, 0.2
nx, perturb = str(int(nx)), str(perturb)

file_base = '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics/SLP/Taylor Green Vortex/PySPH-Testing'


sph_schm = ['00', '01', '02', '03']

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

    if schm == '00':
        plt.semilogy(t, decay_ex, '--k', linewidth=3, label="exact") # Exact Solution

    plt.semilogy(t, decay, linewidth=2, label=schm) # Simulation Solution
    
# Formatting    
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'Max Velocity $\rightarrow$')
tle = r'Max Velocity in flow vs Time'
plt.title(tle)
plt.legend()
tle = file_base + '/decay.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot L_{\infty} error
################################################################################
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
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'$L_\infty$ error $\rightarrow$')
tle = r'$L_\infty$ error vs Time'
plt.title(tle)
plt.legend()
tle = file_base + '/linf_error.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot L_1 error
################################################################################
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
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'$L_1$ error $\rightarrow$')
tle = r'$L_1$ error vs Time'
plt.title(tle)
plt.legend()
tle = file_base + '/l1_error.png' 
plt.savefig(tle, dpi=400)

################################################################################
## Plot L_1 error for p
################################################################################
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, p_l1 = data['t'], data['p_l1']
    
    # Plot
    plt.plot(t, p_l1,linewidth=2, label=schm)
    #plt.semilogy(t, p_l1,linewidth=2, label=schm)
    
# Formatting
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'$L_1$ error for $p \rightarrow$')
tle = r'$L_1$ error for $p$ vs Time'
plt.title(tle)
plt.legend()
tle = file_base + '/p_l1_error.png' 
plt.savefig(tle, dpi=400)



################################################################################
## Run-time
################################################################################
'''
nx_low = [54.58, 38.81, 31.11, 35.25]
nx_high = [98.9, 126.99, 78.52, 97.91]
x_ax = ['dpsph', 'edac', 'tvf', 'wcsph']

sz = (15,10)
plt.figure(figsize=sz)
plt.plot(x_ax, nx_high, 'ob',linewidth=2, label='nx:50')
plt.plot(x_ax, nx_high, '--b',linewidth=2)
plt.plot(x_ax, nx_low, 'ok', linewidth=2, label='nx:30')
plt.plot(x_ax, nx_low, '--k', linewidth=2)
# Formatting
plt.xlabel('SPH Scheme')
plt.ylabel(r'Average runtime (avg[3 runs]) (s) $\rightarrow$')
tle = r'Average runtime (avg[3 runs]) vs SPH Scheme'
plt.title(tle)
plt.legend()
plt.grid()
tle = 'runtime.png' 
plt.savefig(tle, dpi=400)
'''