import numpy as np


logmin = 1e10

#I tend to put all the fitting stuff into a class, it makes it easier to organise and modify things
#Depending on your fitting problem, there can be a lot of bookkeeping involved
class Posterior(object):
    
    #this is all the info the fitting object needs when you instantiate it
    def __init__(self, data, sigma, model, priors, xpix, ypix, par_switches, offsets=None, incset=None, rotset=None):

        #anything stored into self can be used by the other attributes of the object after you instantiate
        self.imsize = data.shape
    
        self.priors = priors #priors lower and upper boundaries
        self.model = model #the image model (in this case that's going to be the wilkin_model above)
        self.data = data
        self.sigma = sigma
        
        #You probably don't want to fit the entire image that you load in, so I've set it up so that you can
        #give a list of the x and y-indices to define which pixels you want to include in the fit
        #(I'll include an example further down below)
        self.xpix = xpix 
        self.ypix = ypix
    
        #because we don't use all pixels in the image and therefore don't have a neat 2d square array,
        #I 'ravel' all the pixels that we want to include into one long 1-d array
        # you then compare with the ravelled model array
        self.ravdata = np.ravel(data[xpix, ypix])
        self.ravsigma = np.ravel(sigma[xpix, ypix])

    
        #switching parameters on and off as fixed parameters
        #these should be True or False
        self.posfix = par_switches[0]
        self.incfix = par_switches[1]
        self.rotfix = par_switches[2]
        
    
        self.offsets = offsets
        self.incset = incset
        self.rotset = rotset
        
        #if offset, inclination, or rotation are fixed, the class expects pre-set values for these
        if self.posfix == True:
            if not self.offsets:
                sys.exit('You set the x and y offset to be fixed. Please pass an offset tuple [x,y] with the offsets keyword when instantiating the GaussianPosterior class')
        if self.incfix == True:
            if not self.incset:
                sys.exit('You set the inclination to be fixed, please pass an inclination with the incset keyword when instantiating the GaussianPosterior class')
        if self.rotfix == True:
            if not self.rotset:
                sys.exit('You set the rotation to be fixed, please pass a rotation with the rotset keyword when instantiating the GaussianPosterior class')

        
    def logprior(self, pars):
        
        r0 = pars[0]
        norm = pars[1]
        bkg = pars[2]
        
        x_off = pars[3]
        y_off = pars[4]
        incl = pars[5]
        rot = pars[6]
        
        #if x_off, y_off, inc and rot are fixed,
        #overwrite them with the fixed values
        if self.posfix == True:
            x_off = self.offsets[0]
            y_off = self.offsets[1]
        
        if self.incfix == True:
            incl = self.incset
        
        if self.rotfix == True:
            rot = self.rotset
            
        # Here you can define priors on the parameters
        # In this ex. i've only included uniform (or 'flat') priors, basically just an upper and a lower boundary

        pb = self.get_priorbounds() #these are the ones getting fed to the class when you instantiate GaussianPosterior
        p_r0 = (r0 > pb[0][0]) & (r0 < pb[0][1])
        p_norm = (norm > pb[1][0])
        p_bkg = (bkg > pb[2][0]) & (bkg < pb[2][1])
        p_xoff = (x_off > pb[3][0]) & (x_off < pb[3][1])
        p_yoff = (y_off > pb[4][0]) & (y_off < pb[4][1])
        p_incl = (incl > pb[5][0]) & (incl < pb[5][1])
        p_rot = (rot  > pb[6][0]) & (rot < pb[6][1])

        logprior = np.log(p_r0 * p_bkg * p_norm * p_incl * p_rot * p_xoff * p_yoff)
    
        #whenever one of the parameters is outside of the boundary, logprior becomes infinite
        #in that case, return the 'logmin' value as defined up above, which tells the algorithm that this
        #parameter value is incredibly unlikely, so it'll be pushed to move that par. back within your prior boundaries
        if not np.isfinite(logprior):
            return logmin
        else:
            return logprior

    def get_priorbounds(self):
        
        return self.priors
    
    def loglikelihood(self, pars):
        
        #This is the likelihood function from Bayesian statistics (i.e., how likely is the data given the model)
        #First get the model values
        model_vals = self.get_ravmodel(pars)

        #equation for the log-likelihood assuming your data points have Gaussian errors
        #(which we are assuming in this case)
        #this is basically equivalent to least-squares regression
        #(i.e. note the (data-model)**2/sigma**2 term at the end)
        ll = len(self.ravdata)/2. * np.log(2*np.pi*np.median(self.ravsigma)**2)  \
                + np.sum( (self.ravdata-model_vals)**2 / (2*self.ravsigma**2))
        
        #this shouldnt happen but if for some reason the LL is infinite, just make it very unlikely
        if not np.isfinite(ll):
            ll = logmin
        return ll
    
    def logposterior(self, pars,  neg=False, verbose=False):
        
        #the posterior is the multiplication of the prior and the likelihood functions (in log-space, the sum)
        #this is the function we are actually trying to optimize/minimize (or sample, in case of MCMC)
        lp = self.logprior(pars)
        ll = self.loglikelihood(pars)
        lpost = lp + ll
        
        if verbose == True:
            print('-----')
            print('logprior:', lp, 'll:', ll, 'lpost:', lpost)
            print('pars:', pars)
        
        #the neg switch makes it so you can either optimize (the log-posterior) or minimize (the negative log-posterior)
        if neg is True:
            return lpost
        else:
            return -lpost       
    
    def get_ravdata(self):
        #return ravelled data
        return self.data, self.sigma
    
    def get_ravmodel(self, pars):
        
        r0 = pars[0]
        norm = pars[1]
        bkg = pars[2]
        
        x_off = pars[3]
        y_off = pars[4]
        incl = pars[5]
        rot = pars[6]
        
            
        #if x_off, y_off, inc and rot are fixed,
        #overwrite them with the set fixed values
        if self.posfix == True:
            x_off = self.offsets[0]
            y_off = self.offsets[1]
        
        if self.incfix == True:
            incl = self.incset
        
        if self.rotfix == True:
            rot = self.rotset
            
        model_arr = self.model(self.imsize, r0, norm, bkg, x_off, y_off, incl, rot)
        ravmodel = model_arr[self.xpix, self.ypix]
    
        return ravmodel
    
    def __call__(self, pars, neg=False, verbose=False):
        return self.logposterior(pars, neg, verbose)
    
    

        
