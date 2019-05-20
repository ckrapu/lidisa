import numpy as np
from numba import jit

@jit
def nandiff(x,y):
    """
    Number of elements which are the same in two arrays,
    not counting NaNs in either.

    Parameters
    ----------
        x : Numpy array.
            Array to be compared against y
        y : Numpy array
            Additional Numpy array of the same shape as x.

    Returns
    -------
        normed_distance : float
            Fraction of elements which are shared in common,
            not taking into account any elements which are NaN in either
            array.

    """
    if np.all(np.isnan(x)):
        return 0.
    num_in_common = np.nansum(x==y)
    num_dif = np.nansum(x!=y)
    frac = 1-num_in_common/(num_dif + num_in_common)
    return frac

@jit
def dsampler(initial,iterations=5,threshold=0.5,radius=4,
             no_update_bands=[],sampling_mode='conditional',max_tries = 10000,
             dfunc=nandiff,output_shape=None):
    """
    Generator function that implements a basic version of direct sampling.
    Both conditional and unconditional sampling are supported.

    Parameters
    ----------
        initial : 2D Numpy array
            Training image used for sampling. If conditional sampling
            is desired, some entries of initial should have NaN values to
            indicate that they are to be filled in by the sampler.
        iterations : int
            Maximum number of iterations over the entire image.
        threshold : float
            Upper limit on distance metric between an existing piece
            of the image and a suggested completion. This distance is normalized
            between 0 and 1.
        radius : int
            Half the width / height of the suggested completions. A larger
            value will lead to bigger blocks of the training image which are stitched
            in to the realization.
        no_update_bands : List of int
            If conditional sampling is used, this lists any categores
            in the realization image which are not to be pasted in. For example, setting
            this to [0,1] means that any update to the realized image which would place
            0 or 1 in the pixel would instead place a NaN (i.e. not update it). This is
            meant to enforce constraints that some categories are fully observed and therefore
            no new values of that category should be added to the realization.
        sampling_mode : string
            Either 'conditional' or 'unconditional'. Unconditional sampling
            starts with a blank image and fills it in while conditional sampling begins
            with an image that is partially filled in and attempts to complete it based
            on the surrounding pixels.
        max_tries : int
            Maximum number of suggested matches to a location before giving up and
            choosing the match which has given the smallest distance so far.
        dfunc : function
            Function of two arguments which returns a number between 0 and 1, with 0 suggesting
            closer similarity and 1 suggestive of greater difference.
        output_shape : List of int
            Sequence of numbers providing the shape of the realized image if
            unconditional sampling is used. If this is not supplied, the image will be
            the same size as the training image.

    Returns
    -------
        realization : Numpy array
            Image which has been filled-in via direct sampling

    """

    xs = np.arange(radius,initial.shape[1]-radius)
    ys = np.arange(radius,initial.shape[0]-radius)

    if sampling_mode == 'conditional':
        # If we are using the same image to sample templates
        # and fill in, we need to specify the sites which
        # can be updated.
        realization    = initial.copy()
        training_image = initial
        pairs          = list(zip(*np.where(np.isnan(training_image))))

    elif sampling_mode == 'unconditional':
        # Alternately, if we want to sample unconditionally,
        # we start with a blank image and iteratively sample
        # from the provided training image
        if output_shape is None:
            output_shape = initial.shape

        realization    = np.zeros_like(output_shape) * np.nan
        training_image = initial
        pairs          = list(product(ys,xs))

    current_iter = 0
    nan_window   = np.zeros([radius*2,radius*2]) * np.nan
    true_window  = np.ones([radius*2,radius*2],dtype=bool)
    false_window = np.zeros_like(true_window,dtype=bool)

    while current_iter < iterations:
        shuffle(pairs)
        for pair in pairs:
            current_window = realization[pair[0]-radius:pair[0]+radius,pair[1]-radius:pair[1]+radius]

            if sampling_mode == 'conditional':
                is_pixel_observed = np.isfinite(initial[pair[0]-radius:pair[0]+radius,pair[1]-radius:pair[1]+radius])
            else:
                is_pixel_observed = false_window

            distance = 1.0
            for attempt in range(max_tries):
                x,y = np.random.choice(xs),np.random.choice(ys)
                candidate  = training_image[y-radius:y+radius,x-radius:x+radius].copy()
                new_distance = dfunc(current_window,candidate)

                if new_distance < threshold:
                    current_best = candidate
                    break

                elif new_distance < distance:
                    distance     = new_distance
                    current_best = candidate

            is_no_update = np.isin(current_best,no_update_bands)
            use_current  = np.logical_or(is_pixel_observed,is_no_update)

            # For a pixel to be included, it must 1) not overwrite an existing observed
            # data value and 2) not be one of the excluded bands
            new_segment = np.select([use_current,true_window],[current_window,current_best])
            realization[pair[0]-radius:pair[0]+radius,pair[1]-radius:pair[1]+radius] = new_segment

        current_iter += 1
        yield realization.copy()
