from numpy import ones_like, mgrid, sqrt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def _derivative(x, t):
    A, a = x
    Anew = A*A*(a**4 - 1)/(a**4 + 1)
    anew = -a*A
    return np.array((Anew, anew))


def _scipy_integrate(y0, tf, dt):
    from scipy.integrate import odeint
    result = odeint(_derivative, y0, [0.0, tf])
    return result[-1]


def _numpy_integrate(y0, tf, dt):
    t = 0.0
    y = y0
    while t <= tf:
        t += dt
        y += dt*_derivative(y, t)
    return y


def exact_solution(tf=0.0075, dt=1e-6, n=101):
    """Exact solution for the locus of the circular patch.

    n is the number of points to find the result at.

    Returns the semi-minor axis, A, pressure, x, y.

    Where x, y are the points corresponding to the ellipse.
    """
    import numpy

    y0 = np.array([100.0, 1.0])

    try:
        from scipy.integrate import odeint
    except ImportError:
        Anew, anew = _numpy_integrate(y0, tf, dt)
    else:
        Anew, anew = _scipy_integrate(y0, tf, dt)

    dadt = _derivative([Anew, anew], tf)[0]
    po = 0.5*-anew**2 * (dadt - Anew**2)

    theta = numpy.linspace(0, 2*numpy.pi, n)

    return anew, Anew, po, anew*numpy.cos(theta), 1/anew*numpy.sin(theta)

file_base = '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics/SLP/Elliptical-Drop/PySPH-Testing'

sph_schm = ['00', '01', '02', '03', '04']

sph_schm_legend = {
    '00': 'WCSPH', '01': 'IISPH', '02': r'$\delta$-SPH', '03': r'$\delta^+$-SPH, GS',
    '04': r'$\delta^+$-SPH, QS'
}
title_additional = ''
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
## Plot Linear Momentum History
################################################################################
plt.figure(figsize=sz)

for schm in sph_schm:        
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, mom = data['t'], data['mom']
    
    # Plot
    plt.plot(t, mom, linewidth=2, label=sph_schm_legend[schm])
    
# Formatting    
plt.xlabel('t')
plt.ylabel('Linear Momentum')
tle = r'Linear Momentum History' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/l_mom_history' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)
################################################################################
## Plot Semi-major axis history
################################################################################
plt.figure(figsize=sz)

cnt = 0
for schm in sph_schm:        
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, ymax = data['t'], data['ymax']
    
    # Plot
    if cnt == 0:
        aa = []
        for tf in t:
            a, A, po, xe, ye = exact_solution(tf)
            aa.append(1.0/a)
        plt.plot(t,aa,'k--',label='exact')
        cnt = 1
    plt.plot(t, ymax, label=sph_schm_legend[schm])
    
# Formatting    
plt.xlabel('t')
plt.ylabel('Semi-Major Axis')
tle = r'Semi-Major Axis History' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/SMA' + savefig_additional + '.png' 
plt.savefig(tle, dpi=400)

################################################################################
## Plot Semi-major axis error
################################################################################
plt.figure(figsize=sz)

for schm in sph_schm:        
    file_loc = file_base + '/Outputs/' + schm + '/results.npz'
    # Read data
    data = np.load(file_loc)
    
    # Unpack
    t, ymax = data['t'], data['ymax']
    
    # Plot
    y = []
    for i in range(len(t)):
        a, A, po, xe, ye = exact_solution(t[i])
        y.append(np.abs(1 - ymax[i]*a)*100.0)


    plt.plot(t, y, label=sph_schm_legend[schm])
    
# Formatting    
plt.xlabel('t')
plt.ylabel('Semi-Major Axis Error (%)')
tle = r'Semi-Major Axis History Error' + title_additional
plt.title(tle)
plt.legend()
tle = file_base + '/SMA_err' + savefig_additional + '.png' 
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
plt.rcParams.update({'font.size': 14})
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