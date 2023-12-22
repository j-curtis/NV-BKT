import numpy as np
import time 
from scipy import integrate as intg 
from scipy import special 
from scipy import stats


### We use units where 
### T_BKT^0 = 1 
### xi_0 = 1
### rho_2d = 2/pi T_BKT^0 = 2/pi

###These parameters govern the solution to the RG flow and are important for various base methods
numrgsteps = 800
lmax = 1.e12  ### Maximum length we integrate to in units of coherence length

dataDirectory = "../data/"


### Filter function appearing in epsilon(q) integral 
### trueForm indicates whether we use the mathematically correct form (derived in paper) which is ill-behaved due to sign-changing oscillations, or an approximate form that is Gaussian
def filterFunction(x,trueForm = False):
	if trueForm:
	    return 2.*special.jv(1,x)/x
	    
	else:
	    return np.exp(-x**2/8.)


### The unrenormalized vortex fugacity using a somewhat arbitrary model of the free energy cost of a vortex at the coherence length scale
def y0(t):
	"""Bare vortex fugacity"""

	return 0.1*np.exp(-1./t)


### RG solver function 
def dxdl(l,x,t):
	"""vectorized RG beta functions"""

	eps = x[0]
	y = x[1]
	return np.array([4.*np.pi**3 * y**2, 2.*(1.-1./(eps*t))*y ])


### Integrates the RG equations and returns a tuple of the length scales and solutions 
def RGSolver(t):
	"""Integrate RG equations and return log(l), (epsilon, y) result as tuple. 
	Integrates up to max length scale by default."""

	logspan = (0.,np.log(lmax))
	x0 = np.array([1.,y0(t)])

	sol = intg.solve_ivp(dxdl,logspan,x0,args=(t,),t_eval=np.linspace(logspan[0],logspan[1],numrgsteps))

	return sol.t,sol.y


### Identifies the integer index where the flow equations should be stopped since y = 1 is reached 
def flowStopIdx(t):
	logls, xs = RGSolver(t)

	### We implement a catch method which will terminate RG flow if fugacity becomes one and use this length scale to compute free vortex density
	ys = xs[1,:]
	xiplus = lmax
	idxMax = len(ys)-1
	if np.any(ys >= 1.):
	    idxMax = np.where(ys >= 1.)[0][0]
	            
	return idxMax


### Identifies the length scale where the flow equations should be stopped since y = 1 is reached 
def xiplus(t):
	logls, xs = RGSolver(t)

	### We implement a catch method which will terminate RG flow if fugacity becomes one and use this length scale to compute free vortex density
	ys = xs[1,:]
	xiplus = lmax
	idxMax = len(ys)-1
	if np.any(ys >= 1.):
	    idxMax = np.where(ys >= 1.)[0][0] 
                
	return np.exp(logls[idxMax])


### Momentum space integrand for noise integral over momentum to get depth dependence
def NVFilter(q,z):
	return q * np.exp(-2.*z*q)


### Function which fits the correlation length divergence to extract critical temperature
def TBKTFit():
	nts = 50
	fitstart = 25

	ts = np.linspace(0.75,1.,nts)

	xis = np.zeros(nts)

	for nt in range(nts):
	    xis[nt] = xiplus(ts[nt]) 
        
	ys = 1./(np.log(xis[fitstart:]))**2
	xs = ts[fitstart:]

	fit = stats.linregress(xs,ys)
	
	m = fit.slope
	b = fit.intercept
	m_sigma = fit.stderr
	b_sigma = fit.intercept_stderr

	TBKT = -b/m 
	TBKT_sigma = b_sigma/np.abs(m) + np.abs(b)/m**2 * m_sigma

	print("Fit: TBKT = "+str(TBKT))
	print("Fit: sigma(TBKT) = "+str(TBKT_sigma))
	print("Fit: r = "+str(fit.rvalue))
	
	return TBKT, TBKT_sigma, fit.rvalue


### Main routine for computing 
def main():
	print("numrgsteps: "+str(numrgsteps))
	print("lmax: "+str(lmax))

	### Parameters which pertain to vortex dynamics
	RN = 0.3 ### kOhm
	RQ = 25.8 ### kOhm
	muv = 4.*RN/RQ ### Bardeen Stephens vortex mobility in terms of RN
	print("muv: "+str(muv))

	### Momentum parameters
	### Momenta integrated over 
	### We use a log sampled scale
	qmin = .1/lmax
	qmax = 2. 
	numqs = 3000

	qs = np.geomspace(qmin,qmax,numqs)
	np.save(dataDirectory+"qs.npy",qs)

	### Frequency parameters
	### Frequencies relative to diffusive frequency scale on log scale
	#numws = 10 
	ws = muv*np.array([1.e-25,1.e-8, 3.e-8, 1.e-7, 3.e-7, 1.e-6, 3.e-6, 1.e-5, 3.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.])
	numws = len(ws)
	print("numws: "+str(numws))
	np.save(dataDirectory+"ws.npy",ws)

	### Extract true transition temperature
	TBKT, TBKT_sigma, r = TBKTFit()

	### Temperature parameters
	numtemps = 40
	temps = np.linspace(0.8*TBKT,1.3*TBKT,numtemps)
	np.save(dataDirectory+"temps.npy",temps)

	### Now we form the solutions to the flow equations
	### For each temperature there will be an array of size numrgsteps
	logls = np.zeros(numrgsteps)	### Length scales in rg calculation -- should be same for all temperatures
	eps_l = np.zeros((numtemps,numrgsteps))	### epsilons as a function of length scale for each temperature
	y_l = np.zeros((numtemps,numrgsteps))	### fugacity as a function of length scale for each temperature
	
	xis = np.zeros(numtemps)	### xi_+ length scale (if finite) for each temperature
	nfs = np.zeros(numtemps)	### free vortex density (if nonzero) for each temperature

	eps_b_qw = np.zeros((numtemps,numqs,numws),dtype=complex) ### Vortex dielectric functions for bound pair contribution
	eps_f_qw = np.zeros((numtemps,numqs,numws),dtype=complex) ### Vortex dielectric functions for free pair contribution
	eps_qw = np.zeros((numtemps,numqs,numws),dtype=complex) ### Total vortex dielectric functions 


	#numzs = 20
	#zs = np.logspace(0.,3.,numzs)
	zs = np.array([1.,3.,5.,10.,30.,50.,100.,300.,500.,1000.,3000.])
	numzs = len(zs)
	print("numzs: "+str(numzs))
	np.save(dataDirectory+"zs.npy",zs)

	noise = np.zeros((numtemps,numzs,numws))

	t0 = time.time()
	for nt in range(numtemps):
		t = temps[nt]
		D = muv*t

		logls, y = RGSolver(t)
		eps_l[nt,:] = y[0,:]
		y_l[nt,:] = y[1,:]

		idxMax = flowStopIdx(temps[nt])
		xis[nt] = np.exp(logls[idxMax])

		if temps[nt] > TBKT:
		#if xis[nt] < lmax:
			nfs[nt] = 1./(np.pi*xis[nt]**2)

		### Now we form the integrand kernel for the dielectric function 

		depsdlogl = 4.*np.pi**3*y_l[nt,:]**2
		weight_function = np.zeros((numrgsteps,numqs,numqs),dtype=complex)

		for nq in range(numqs):
			q = qs[nq]

			for nw in range(numws):
				w = ws[nw]

				weight_function[:,nq,nw] = filterFunction(.5*q*np.exp(logls[:]))/(1. - 1.j*w*np.exp(2.*logls[:])/(14.*D) )

				eps_f_qw[nt,nq,nw] = 4.*np.pi**2*2./np.pi*muv*nfs[nt]/(D*q**2 - 1.j*w)
				eps_b_qw[nt,nq,nw] = 1.+ np.trapz(weight_function[:idxMax,nq,nw]*depsdlogl[:idxMax],logls[:idxMax],axis=0)
				eps_qw = eps_f_qw + eps_b_qw


		for nz in range(numzs):
			z = zs[nz]

			noisekernel = -np.outer(NVFilter(qs,z),1./ws)*D 

			noise[nt,nz,:] = np.trapz(noisekernel*np.imag(1./eps_qw[nt,:,:]),qs,axis=0)


	t1 = time.time()

	print("Total loop time: "+str(t1-t0)+"s")

	### Now we save the outputs
	np.save(dataDirectory+"logls.npy",logls)
	np.save(dataDirectory+"eps_l_vs_tlogl.npy",eps_l)
	np.save(dataDirectory+"y_l_vs_tlogl.npy",y_l)

	np.save(dataDirectory+"xis_vs_t.npy",xis)
	np.save(dataDirectory+"nfs_vs_t.npy",nfs)

	np.save(dataDirectory+"eps_b_vs_tqw.npy",eps_b_qw)
	np.save(dataDirectory+"eps_f_vs_tqw.npy",eps_f_qw)
	np.save(dataDirectory+"eps_vs_tqw.npy",eps_qw)

	np.save(dataDirectory+"noise_vs_tzw.npy",noise)




if __name__ == "__main__":
	main()














