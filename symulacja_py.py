import numpy as np
from scipy import signal
from scipy.stats import levy_stable

def impsim(fs,nx,fmod,amp_imp,f_center,bandwidth,shift):
    tp = np.arange(0,0.05+10**(-7),1/fs)
    pnx = len(tp)
    t = np.arange(1,nx+1)/fs
    bpFilt = signal.firwin(81,[f_center-bandwidth,f_center+bandwidth], pass_zero=False, fs = fs)
    syg_c = np.zeros((nx,))
    fault_samples = np.round(fs/fmod)
    gdzie = np.arange(1,(np.ceil(nx/fault_samples))*fault_samples+1,fault_samples,int)
    gdzie = gdzie[gdzie+pnx+1<=len(t)]  
    y = amp_imp*np.sin(2*np.pi*f_center*tp)*np.exp(-3000*tp)
    for i in range(len(gdzie)):
        syg_c[(gdzie[i]-1):(gdzie[i]+pnx-1)] += y
    yy = signal.lfilter(bpFilt,[1],syg_c)
    yy = np.roll(yy,shift) #if shift
    return yy


fs = 25000
fmod = 30
f_center = 5000
bandwidth = 1500
shift = 0

amp_imp = 1
alfa = 1.8
skala = 0.5

varsize = 1*fs
tt_ts = np.linspace(1/fs,varsize/fs,varsize)
#szum = np.random.normal(0,skala,varsize) #szum Gaussowski
szum = levy_stable.rvs(alfa,0,0,skala,varsize)  #szum alfa-stabilny

signal_l = 2*impsim(fs,varsize,fmod,amp_imp,f_center,bandwidth,shift)+szum
