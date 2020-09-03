import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


#nx, perturb = 50, 0.2
#nx, perturb = str(int(nx)), str(perturb)

file_base = '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics/SLP/Taylor Green Vortex/PySPH-Testing/'


sph_schm = ['08', '03', '16', '17', '14', '18', '19']
sph_schm = ['14', '18', '19', '17', '08']

sph_schm_legend = {
    '00':  r'$\delta$-SPH', '01': r'$\delta$-SPH, Prt', '02': 'DPSPH-SPH', 
    '03': 'DPSPH, Prt', '04': 'Case 1', '05': 'Case 2',
    '06': r'$\delta^+$-SPH', '07': r'$\delta^+$-SPH, Prt, QS, hdx=1', '08': 'EDAC, Prt',
    '09': 'EDAC', '10': r'$\delta^+$-SPH, Prt, TEOS', 
    '11': r'$\delta^+$-SPH, Prt, nx=30',
    '12': r'$\delta^+$-SPH, Prt, nx=30', '13': r'$\delta$-SPH, Prt, nx=30', 
    '14': 'EDAC, Prt, nx=30', '15':  r'$\delta^+$-SPH, Prt, WQK, hdx=2',
    '16': r'$\delta^+$-SPH, Prt, WQK, hdx=1.33', 
    '17': r'$\delta^+$-SPH, Prt, WQK, hdx=1.33, PST',
    '18': r'$\delta^+$-SPH, nx=30, Prt, WQK, hdx=1.33, PST',
    '19': r'$\delta^+$-SPH, nx=30, Prt, WQK, hdx=1.33',
    '20': 'Check1', '21': 'Check2'
    }
title_additional = ' | nx = 50 | Perturb = 0.2'
savefig_additional = ''

################################################################################
## Plot decay
################################################################################
sz = (19.20,10.80)
plt.figure(figsize=sz)

cnt = 1
for schm in sph_schm:        
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, decay, decay_ex = data['t'], data['decay'], data['decay_ex']
    
    # Plot

    if cnt == 1:
        plt.semilogy(t, decay_ex, '--k', linewidth=3, label="exact") # Exact Solution
        cnt = 0

    plt.semilogy(t, decay, linewidth=2, label=sph_schm_legend[schm]) # Simulation Solution
    
# Formatting    
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'Max Velocity $\rightarrow$')
tle = r'Max Velocity in flow vs Time' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/decay' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot L_{\infty} error
################################################################################
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, linf = data['t'], data['linf']
    
    # Plot
    plt.plot(t, linf, linewidth=2, label=sph_schm_legend[schm])
    
# Formatting
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'$L_\infty$ error $\rightarrow$')
tle = r'$L_\infty$ error vs Time' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/linf_error' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Plot L_1 error
################################################################################
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, l1 = data['t'], data['l1']
    
    # Plot
    plt.plot(t, l1,linewidth=2, label=sph_schm_legend[schm])
    
# Formatting
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'$L_1$ error $\rightarrow$')
tle = r'$L_1$ error vs Time' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/l1_error' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)

################################################################################
## Plot L_1 error for p
################################################################################
plt.figure(figsize=sz)

for schm in sph_schm:
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, p_l1 = data['t'], data['p_l1']
    
    # Plot
    plt.plot(t, p_l1,linewidth=2, label=sph_schm_legend[schm])
    #plt.semilogy(t, p_l1,linewidth=2, label=schm)
    
# Formatting
plt.xlabel(r't $\rightarrow$')
plt.ylabel(r'$L_1$ error for $p \rightarrow$')
tle = r'$L_1$ error for $p$ vs Time' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/p_l1_error' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)


################################################################################
## Run-time
################################################################################
def extract_RT(file_loc):

    from os import walk
    files = []
    for (dirpath, dirnames, filenames) in walk(file_loc):
        files.extend(filenames)
        break

    fname = ''
    for i in files:
        if i.endswith('.log'):
            fname = i

    file_loc += '/' + fname
    data = open(file_loc, 'r')
    lines = data.read()
    rt = float(lines[lines.find('Run took: ')+10:].split(' secs')[0])
    data.close()

    return rt


RT_y = []
RT_x = []
for schm in sph_schm:
    file_loc = file_base + '/Outputs/' + schm 
    # Read data
    RT_y.append(extract_RT(file_loc))
    RT_x.append(sph_schm_legend[schm])
    
# Plotting
plt.rcParams.update({'font.size': 10})
sz = (22,10.8)
fig, ax = plt.subplots(figsize=sz)

# Horizontal Bar Plot 
ax.barh(RT_x, RT_y, height=0.4)
# Remove axes splines 
for s in ['top', 'bottom', 'left', 'right']: 
    ax.spines[s].set_visible(False) 
  
# Remove x, y Ticks 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 

ax.set_xlabel(r'Run time (second) $\rightarrow$')
  
# Add padding between axes and labels 
#ax.xaxis.set_tick_params(pad = 5) 
#ax.yaxis.set_tick_params(pad = 10) 
  
# Add x, y gridlines 
ax.grid(b = True, color ='black', 
        linestyle ='-.', linewidth = 0.7, 
        alpha = 0.2) 
  
# Show top values  
#ax.invert_yaxis() 
  
# Add annotation to bars 
for i in ax.patches: 
    plt.text(i.get_width(), i.get_y()+0.2,  
             str(round((i.get_width()), 2)), 
             fontsize = 11, fontweight ='bold', 
             color ='grey') 
  
# Add Plot Title 
ax.set_title('Run Time - SPH Schemes') 
tle = file_base + '/run_time' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)