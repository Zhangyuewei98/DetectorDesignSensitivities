import astropy.units as u
import numpy as np

#########Empirical Distributions########

# class used to define a 1D empirical distribution
# based on posterior from another MCMC
class EmpiricalDistribution1D(object):

    def __init__(self, param_name, samples, bins):
        """
            :param samples: samples for hist
            :param bins: edges to use for hist (left and right)
            make sure bins cover whole prior!
            """
        self.ndim = 1
        self.param_name = param_name
        self._Nbins = len(bins)-1
        hist, x_bins = np.histogram(samples, bins=bins)

        self._edges = x_bins[:-1]
        self._wids = np.diff(x_bins)

        hist += 1  # add a sample to every bin
        counts = np.sum(hist)
        self._pdf = hist / float(counts) / self._wids
        self._cdf = np.cumsum((self._pdf*self._wids).ravel())

        self._logpdf = np.log(self._pdf)

    def draw(self):
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw)

        idx = np.unravel_index(draw_bin, self._Nbins)
        samp = self._edges[idx] + self._wids[idx]*np.random.rand()
        return np.array(samp)

    def prob(self, params):
        ix = min(np.searchsorted(self._edges, params),
                 self._Nbins-1)

        return self._pdf[ix]

    def logprob(self, params):
        ix = min(np.searchsorted(self._edges, params),
                 self._Nbins-1)

        return self._logpdf[ix]


# class used to define a 2D empirical distribution
# based on posteriors from another MCMC
class EmpiricalDistribution2D(object):
    def __init__(self, param_names, samples, bins):
        """
            :param samples: samples for hist
            :param bins: edges to use for hist (left and right)
            make sure bins cover whole prior!
            """
        self.ndim = 2
        self.param_names = param_names
        self._Nbins = [len(b)-1 for b in bins]
        hist, x_bins, y_bins = np.histogram2d(*samples, bins=bins)

        self._edges = np.array([x_bins[:-1], y_bins[:-1]])
        self._wids = np.diff([x_bins, y_bins])

        area = np.outer(*self._wids)
        hist += 1  # add a sample to every bin
        counts = np.sum(hist)
        self._pdf = hist / counts / area
        self._cdf = np.cumsum((self._pdf*area).ravel())

        self._logpdf = np.log(self._pdf)

    def draw(self):
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw)

        idx = np.unravel_index(draw_bin, self._Nbins)
        samp = [self._edges[ii, idx[ii]] + self._wids[ii, idx[ii]]*np.random.rand()
                for ii in range(2)]
        return np.array(samp)

    def prob(self, params):
        ix, iy = [min(np.searchsorted(self._edges[ii], params[ii]),
                      self._Nbins[ii]-1) for ii in range(2)]

        return self._pdf[ix, iy]

    def logprob(self, params):
        ix, iy = [min(np.searchsorted(self._edges[ii], params[ii]),
                      self._Nbins[ii]-1) for ii in range(2)]

        return self._logpdf[ix, iy]


def make_empirical_distributions(paramlist, params, chain,
                                 burn=0, nbins=50):
    """
        Utility function to construct empirical distributions.
        :param paramlist: a list of parameter names,
                          either single parameters or pairs of parameters
        :param params: list of all parameter names for the MCMC chain
        :param chain: MCMC chain from a previous run
        :param burn: desired number of initial samples to discard
        :param nbins: number of bins to use for the empirical distributions
        :return distr: list of empirical distributions
        """

    distr = []

    for pl in paramlist:

        if type(pl) is not list:

            pl = [pl]

        if len(pl) == 1:

            # get the parameter index
            idx = params.index(pl[0])

            # get the bins for the histogram
            bins = np.linspace(min(chain[burn:, idx]), max(chain[burn:, idx]), nbins)

            new_distr = EmpiricalDistribution1D(pl[0], chain[burn:, idx], bins)

            distr.append(new_distr)

        elif len(pl) == 2:

            # get the parameter indices
            idx = [params.index(pl1) for pl1 in pl]

            # get the bins for the histogram
            bins = [np.linspace(min(chain[burn:, i]), max(chain[burn:, i]), nbins) for i in idx]

            new_distr = EmpiricalDistribution2D(pl, chain[burn:, idx].T, bins)

            distr.append(new_distr)

        else:
            print('Warning: only 1D and 2D empirical distributions are currently allowed.')

def make_quant(param, default_unit):
    """Convenience function to intialize a parameter as an astropy quantity.

    Parameters
    ----------
    param : float, or Astropy Quantity
        Parameter to initialize
    default_unit : str
        Astropy unit string, sets as default for param.

    Returns
    -------
        an astropy quantity
    
    Examples
    --------
        self.f0 = make_quant(f0,'MHz')

    Notes
    -----
    Taken from <https://github.com/Hazboun6/hasasia/blob/master/hasasia/sensitivity.py#L834>

    """
    default_unit = u.core.Unit(default_unit)
    if hasattr(param, 'unit'):
        try:
            quantity = param.to(default_unit)
        except u.UnitConversionError:
            raise ValueError("Quantity {0} with incompatible unit {1}"
                             .format(param, default_unit))
    else:
        quantity = param * default_unit

    return quantity


def Get_Var_Dict(obj,value):
    """Updates and initializes variable dictionaries used to keep track of
    current values and variable minima and maxima.

    Parameters
    ----------
    obj : object
        Instance of class with parameter variables
    value : array-like
        value(s) that are assigned into dictionary

    Notes
    -----
    value contains the variable name in the first index
    the next is the current value of the variable
    the last two are optional and contain the variable min and max

    Examples
    --------
    obj.var_dict = ['M',value]
        where obj is in this case an instance of a BinaryBlackHole

    """
    if not hasattr(obj,'var_dict'):
            obj._var_dict = {}
    if isinstance(value,list):
        if len(value) == 2 and isinstance(value[0],str):
            var_name = value[0]
            vals = value[1]
            if isinstance(vals,list) and len(vals) == 3:
                if isinstance(vals[0],(float,int,u.Quantity))\
                 and isinstance(vals[1],(float,int,u.Quantity))\
                  and isinstance(vals[2],(float,int,u.Quantity)):
                    obj._return_value = vals[0]
                    obj._var_dict[var_name] = {'val':vals[0],'min':vals[1],'max':vals[2]}
                else:
                    raise ValueError(DictError_3())
            elif isinstance(vals,(float,int,u.Quantity)):
                if isinstance(vals,(float,int,u.Quantity)):
                    if var_name in obj._var_dict.keys():
                        obj._var_dict[var_name]['val'] = vals
                    else:
                        obj.var_dict[var_name] = {'val':vals,'min':None,'max':None}
                    obj._return_value = vals
                else:
                    raise ValueError(DictError_2())
        else:
            raise ValueError(DictError_Full())
    else:
        raise ValueError(DictError_Full())

def DictError_Full():
    return 'Must assign either: \n\
    - A name and value in a list (ie. ["name",val]), or \n\
    - A name, a value, a minimum value, and maximum value in a list (ie. ["name",val,min,max]), \n\
    where where name is a string, and val,min,and max are either floats, ints, or an astropy Quantity.'
def DictError_3():
    return 'Must assign a name, a value, a minimum value, and maximum value in a list (ie. ["name",val,min,max]), \n\
    where name is a string, and val, min, and max are either floats, ints, or astropy Quantities.'
def DictError_2():
    return 'Must assign a name and value in a list (ie. ["name",val]) \n\
    where name is a string, and val is either a float, an int, or an astropy Quantity.'