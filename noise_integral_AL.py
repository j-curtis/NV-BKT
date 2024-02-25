import numpy as np
import time 
from scipy import integrate as intg 
from matplotlib import pyplot as plt
from matplotlib import cm

### In this code we evaluate the integral for the noise from Aslamazov-Larkin current fluctuations

### We use units where 
### T_c = 1 
### xi_0 = 1

dataDirectory = "../data/AL/"
figDirectory = "../figures/AL/"


### Main routine for computing 
def compute():

	### Integration parameters
	xmax = 6.
	vmax = 150.
	
	###	Plot parameters
	numrs = 20
	rmin = 1.e-3
	rmax = 1.
	### We avoid getting too close to the critical point
	rs = np.geomspace(rmin,rmax,numrs)
	#rs = np.array([1.e-3,3.e-3,5.e-3,1.e-2,3.e-2,5.e-2,1.e-1,3.e-1,5.e-1,1.e0])
	#numrs = len(rs)

	zs = np.array([1.,3.,5.,10.,30.,50.,100.,300.])
	numzs = len(zs)

	noise = np.zeros((numrs,numzs))
	noise_err = np.zeros_like(noise)

	for nr in range(numrs):
		r = rs[nr]

		for nz in range(numzs):
			z = zs[nz]

			#def integrand(x,u):
			def integrand(x,v):
				"""Integrand in AL noise formula"""
				return .5*(1.+r)**2*np.exp(-4.*x*z)/x *( (x**2 + v + r) - np.sqrt( (x**2 + r + v)**2 - 4.*v*x**2 ) )/( (x**2 + v + r)**2 )

				#return (1.+r)**2*np.exp(-4.*x*z)/x *( (x**2 + u**2 + r) - np.sqrt( ( (x+u)**2 + r )*( (x-u)**2 + r ) ) )/( (x**2 + u**2 + r)**2 )



			#noise[nr,nz], noise_err[nr,nz] = intg.dblquad(integrand, 0.,umax,0.,xmax)
			noise[nr,nz], noise_err[nr,nz] = intg.dblquad(integrand, 0.,vmax,0.,xmax)


	np.save(dataDirectory+"noise.npy",noise)
	np.save(dataDirectory+"noise_err.npy",noise_err)
	np.save(dataDirectory+"zs.npy",zs)
	np.save(dataDirectory+"rs.npy",rs)


def anaylyze(saveFig):

	#plt.rc('figure', dpi=300)
	plt.rc('font', family = 'Times New Roman')
	plt.rc('font', size = 14)
	plt.rc('text', usetex=True)
	plt.rc('xtick', labelsize=14)
	plt.rc('ytick', labelsize=14)
	plt.rc('axes', labelsize=18)
	plt.rc('lines', linewidth=2.)
	plt.rc('lines',marker='o')

	zs = np.load(dataDirectory+"zs.npy")
	rs = np.load(dataDirectory+"rs.npy")
	noise = np.load(dataDirectory+"noise.npy")
	noise_err = np.load(dataDirectory+"noise_err.npy")

	zindxs = [3,4,6,7]
	cs = cm.gist_heat(np.linspace(0.8,0.,len(zindxs)))

	i = 0
	for zindx in zindxs:
		plt.plot(rs,noise[:,zindx],color=cs[i],label=r'$z = $'+"{z:0.0f}".format(z=zs[zindx])+r'$\xi_0$' )
		i+=1

	plt.legend()
	plt.xlabel(r'$r$')
	plt.ylabel(r'$\mathcal{N}_{zz}/\mathcal{N}_0^{\rm AL}$')
	plt.yscale('log')
	plt.xscale('log')

	if saveFig:
	    plt.savefig(figDirectory+"noise_vs_temp.pdf",bbox_inches='tight')

	plt.show()

	rindxs = [0,6,12,19]
	cs = cm.coolwarm(np.linspace(0.55,1.,len(rindxs)))

	i = 0
	for rindx in rindxs:
		plt.plot(zs,noise[rindx,:],color=cs[i],label=r'$r = $'+"{r:0.04E}".format(r=rs[rindx]) )
		i+=1

	plt.legend()
	plt.xlabel(r'$z/\xi_0$')
	plt.ylabel(r'$\mathcal{N}_{zz}/\mathcal{N}_0^{\rm AL}$')
	plt.yscale('log')
	plt.xscale('log')

	if saveFig:
	    plt.savefig(figDirectory+"noise_vs_z.pdf",bbox_inches='tight')

	plt.show()

def main():
	saveFig = True
	calc = True

	if calc:
		compute()

	anaylyze(saveFig)

if __name__ == "__main__":
	main()














