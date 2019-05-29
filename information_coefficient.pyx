# cython: infer_types=True
# cython: profile=True
# New version with template kernel density
########
# removing as much numpy as posible

cimport cython

cimport numpy as np
import numpy as np
cdef np.double_t EPS = np.finfo(float).eps

import pandas as pd

from libc.float cimport DBL_EPSILON
from libc.limits cimport INT_MAX
from libc.math cimport exp, log, sqrt, M_PI, fabs, isnan, fmin, fmax, round, isnan
from libc.stdlib cimport rand, malloc, free, srand
from libc.stdio cimport printf

cdef extern from "limits.h":
    int RAND_MAX
cdef int RANDOM_SEED = 20121020

cdef struct density:
    np.double_t d
    np.uint32_t* cnt

cdef np.double_t DELMAX=1000

def convertable_list(y):
    return isinstance(y, list) and (isinstance(y[0], np.ndarray) or isinstance(y[0], pd.Series))

def is_list_of_nparray(y):
    return isinstance(y, list) and isinstance(y[0], np.ndarray)

def is_2darray(y):
    return isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1

def compute_information_coefficient(x, y, n_grids=25, jitter=1E-10, random_seed=RANDOM_SEED, bandwidth_mult=1.0, k=1):
    '''
    Compute the information coefficient between x and ys, which can be composed of either continuous,
    categorical, or binary values.
    :param x: numpy array or array-like; A 1 dimensional array of values
    :param y: numpy array or array-like; A 1 or 2 dimensional array of values
    :param n_grids: int; number of grid lines in a dimention when estimating bandwidths
    :param jitter: number;
    :param random_seed: int or array-like;
    :param k: int; The size of the template
    :return: float or list;
    '''
    cdef np.ndarray[np.double_t, ndim=1] x_array = np.asarray(x)
    cdef np.double_t bcv_x = py_bcv(&x_array[0], x_array.shape[0], x_array.min(), x_array.max()) / 4
    
    if convertable_list(y):
        return [cy_information_coefficient_template_with_bandwidth(x_array, np.asarray(y[i]), n_grids=n_grids, 
                                                                                jitter=jitter, 
                                                                                random_seed=random_seed, 
                                                                                bandwidth_mult=bandwidth_mult, k=k)
                for i in range(len(y))]
    y_array = np.asarray(y)
    if is_2darray(y_array):
        return [cy_information_coefficient_template_with_bandwidth(x_array, y_array[:, i], bcv_x=bcv_x, n_grids=n_grids,
                                                                   jitter=jitter, random_seed=random_seed,
                                                                   bandwidth_mult=bandwidth_mult, k=k) 
                for i in range(y_array.shape[1])]
    else:
        return cy_information_coefficient_template_with_bandwidth(x_array, y_array, bcv_x=bcv_x, n_grids=n_grids, 
                                                                  jitter=jitter, random_seed=random_seed, 
                                                                  bandwidth_mult=bandwidth_mult, k=k)

#'''
@cython.profile(True)
@cython.linetrace(True)
@cython.binding(True)#'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_information_coefficient_between_1_and_multiple_1d_arrays(np.ndarray[np.double_t, ndim=1] x, ys, **kwargs):
    cdef np.double_t bcv_x = py_bcv(&x[0], x.shape[0], x.min(), x.max()) / 4
    if isinstance(ys, np.ndarray):
        return [cy_information_coefficient_template_with_bandwidth(x, ys[:, i], bcv_x=bcv_x, **kwargs) for i in \
                range(ys.shape[1])]
    return [cy_information_coefficient_template_with_bandwidth(x, y, bcv_x=bcv_x, **kwargs) for y in ys]
    

#'''
@cython.profile(True)
@cython.linetrace(True)
@cython.binding(True)#'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_information_coefficient_template_with_bandwidth(np.ndarray[np.double_t, ndim=1] x_in, np.ndarray[np.double_t, ndim=1] y_in,
                                       np.double_t bcv_x=-1, np.double_t bcv_y=-1,
                                       int n_grids=25, np.double_t jitter=1E-10, 
                                       int random_seed=RANDOM_SEED, np.double_t bandwidth_mult=1,
                                       int k=1):
    """
    Compute the information coefficient between x and y, which can be composed of either continuous,
    categorical, or binary values.
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grid lines in a dimention when estimating bandwidths
    :param jitter: number;
    :param random_seed: int or array-like;
    :param k: int; The size of the template
    :return: float;
    """
    cdef np.double_t cor, p, bandwidth_x, bandwidth_y
    cdef np.double_t total_bandwidth_multiplier
    cdef np.double_t x_min, x_max, y_min, y_max, dx, dy, fxy_sum, mi, ic
    cdef np.double_t* fxy
    cdef np.double_t* px
    cdef np.double_t* py
    cdef int i, j
    cdef int len_x = x_in.shape[0]
    cdef np.double_t* x = &x_in[0]
    cdef np.double_t* y = &y_in[0]
    cdef int size = 0
    cdef int temp

    with nogil:
        for i in range(len_x):
            if not (isnan(x[i]) | isnan(y[i])):
                if i != size:
                    x[size] = x[i]
                    y[size] = y[i]
                size += 1

        # Need at least 3 values to compute bandwidth
        if size < 3:
            with gil:
                return 0

        srand(random_seed)
        for i in range(size):
            x[i] += (rand()/<np.double_t>RAND_MAX) * jitter
            y[i] += (rand()/<np.double_t>RAND_MAX) * jitter
        
        if size % 2 == 0:
            if x[0] < x[1]:
                x_min = x[0]
                x_max = x[1]
            else:
                x_min = x[1]
                x_max = x[0]
            if y[0] < y[1]:
                y_min = y[0]
                y_max = y[1]
            else:
                y_min = y[1]
                y_max = y[0]
            i = 2
        else:
            x_min = x[0]
            x_max = x[0]
            y_min = y[0]
            y_max = y[0]
            i = 1
            
        while i < size - 1:
            if x[i] > x[i+1]:
                if x[i] > x_max:
                    x_max = x[i]
                if x[i+1] < x_min:
                    x_min = x[i+1]
            else:
                if x[i+1] > x_max:
                    x_max = x[i+1]
                if x[i] < x_min:
                    x_min = x[i]
            if y[i] > y[i+1]:
                if y[i] > y_max:
                    y_max = y[i]
                if y[i+1] < y_min:
                    y_min = y[i+1]
            else:
                if y[i+1] > y_max:
                    y_max = y[i+1]
                if y[i] < y_min:
                    y_min = y[i]
            i += 2

        # Compute bandwidths
        cor = pearsonr(x, y, size)
        total_bandwidth_multiplier = bandwidth_mult * (1.0 + (-0.75) * fabs(cor))
        if bcv_x == -1:
            bcv_x = py_bcv(x, size, x_min, x_max) / 4
        if bcv_y == -1:
            bcv_y = py_bcv(y, size, y_min, y_max) / 4
        bandwidth_x = total_bandwidth_multiplier * bcv_x
        bandwidth_y = total_bandwidth_multiplier * bcv_y

        fxy = <np.double_t*> malloc(n_grids * n_grids * sizeof(np.double_t))
        for i in range(n_grids * n_grids):
            fxy[i] = 0
        cy_compute_kernel_density(x, y, bandwidth_x, bandwidth_y, size, n_grids - 2*k, k, fxy,
                                  n_grids, x_min, x_max, y_min, y_max)
               
        # Compute P(x, y), P(x), P(y)
        dx = (x_max - x_min) / (n_grids - 1)
        dy = (y_max - y_min) / (n_grids - 1)
        fxy_sum = 0
        for i in range(n_grids * n_grids):
            fxy[i] += EPS
            fxy_sum += fxy[i]
        for i in range(n_grids * n_grids):
            fxy[i] = fxy[i] / (fxy_sum * dx * dy)
        px = <np.double_t*> malloc(n_grids * sizeof(np.double_t))
        py = <np.double_t*> malloc(n_grids * sizeof(np.double_t))
        for i in range(n_grids):
            px[i] = fxy[i * n_grids]
            py[i] = fxy[i]
            for j in range(1, n_grids):
                px[i] += fxy[i * n_grids + j]
                py[i] += fxy[j * n_grids + i]
            px[i] *= dy
            py[i] *= dx

        mi = 0
        for i in range(n_grids):
            for j in range(n_grids):
                mi += fxy[i * n_grids + j] * log(fxy[i * n_grids + j] / (px[i] * py[j]))
        mi = mi * dx * dy

        free(px)
        free(py)
        free(fxy)

        # Compute information coefficient
        ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

        # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
        if isnan(ic):
            ic = 0

        with gil:
            return ic

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline np.double_t pearsonr(np.double_t* x, np.double_t* y, int n) nogil:
    cdef np.double_t mx, my, r_num = 0, r_den, r
    cdef np.double_t sum_x = 0, sum_y = 0
    cdef np.double_t xm, ym
    cdef int i
    mx = x[0]
    my = y[0]
    for i in range(1, n):
        mx += x[i]
        my += y[i]
    mx = mx / n
    my = my / n
    
    for i in range(n):
        xm = x[i] - mx
        ym = y[i] - my
        r_num += xm * ym
        sum_x += xm ** 2
        sum_y += ym ** 2
    r_den = sqrt(sum_x * sum_y)
    r = r_num / r_den
    return r

def test_kern_den(np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y, b_x, b_y, n_grids, k):
    cdef np.ndarray[np.double_t, ndim=2] grid = np.zeros((n_grids, n_grids))
    cy_compute_kernel_density(&x[0], &y[0], b_x, b_y, x.shape[0], n_grids - 2 * k, k, &grid[0,0], n_grids, 
                              x.min(), x.max(), y.min(), y.max())
    return grid

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void cy_compute_kernel_density(np.double_t* x, np.double_t* y, np.double_t b_x, np.double_t b_y, 
                                    int n, int n_grids_2k, int k, np.double_t* grid, int n_grids, 
                                    np.double_t x_min, np.double_t x_max, np.double_t y_min, 
                                    np.double_t y_max) nogil:

    cdef int i, j, d, h, grid_size, x_d, y_d
    
    cdef np.double_t dx = (x_max - x_min)/(n_grids)
    cdef np.double_t dy = (y_max - y_min)/(n_grids)
    cdef np.double_t sigma_x = b_x/dx
    cdef np.double_t sigma_y = b_y/dy
    cdef np.double_t term_2_pi_sigma_x_y = 2 * 3.1415926 * sigma_x * sigma_y 
    
    # define template
    cdef int kern_size = 2*k + 1
    cdef np.double_t* kern_temp = <np.double_t*> malloc((kern_size * kern_size) * sizeof(np.double_t))
    
    h = 0
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            kern_temp[h] = exp(-0.5*((i/sigma_x)**2 + (j/sigma_y)**2))/term_2_pi_sigma_x_y
            h += 1
    
    # rescale samples to grid coordinates
    cdef int index 
    for d in range(n): 
        h = 0
        x_d = rint((x[d] - x_min)/dx) + k
        y_d = rint((y[d] - y_min)/dy) + k
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                index = (x_d + i) * n_grids + y_d + j
                if index >= 0 and index < n_grids * n_grids:
                    grid[index] += kern_temp[h]
                h += 1
        
    free(kern_temp)      

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline density VR_den_bin(int n, int nb, np.double_t* x, np.double_t x_min, np.double_t x_max) nogil:
    cdef int i, j, ii, jj, iij
    cdef density ret
    ret.cnt = <np.uint32_t*> malloc(nb * sizeof(np.uint32_t))
    for i in range(nb):
        ret.cnt[i] = 0
    
    cdef np.double_t rang = (x_max - x_min) * 1.01
    ret.d = rang / nb

    cdef np.double_t x_i, x_j
    for i in range(1,n):
        ii = (int)(x[i] / ret.d)
        for j in range(i):
            jj = (int)(x[j] / ret.d)
            iij = (ii - jj)
            iij = iij if iij >= 0 else -iij
            if(ret.cnt[iij] == INT_MAX):
                printf("maximum count exceeded in pairwise distance binning\n")
            ret.cnt[iij] += 1
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline np.double_t VR_bcv_bin(int n, int nb, density den, np.double_t h) nogil:
    cdef np.double_t total_sum = 0.0
    cdef np.double_t delta, term
    cdef np.double_t hh = h / 4
    cdef int i
    for i in range(nb):
        delta = (i * den.d / hh) 
        delta *= delta
        if delta >= DELMAX:
            break
        term = exp(-1 * delta / 4) * (delta * delta - 12 * delta + 12)
        total_sum += term * den.cnt[i]
    return 1 / (2.0 * n * hh * sqrt(M_PI)) + total_sum / (64.0 * n * n * hh * sqrt(M_PI))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline np.double_t r_var(np.double_t* x, int n) nogil:
    cdef np.double_t mean = 0
    cdef int i
    for i in range(n):
        mean += x[i]
    mean = mean / n
    
    cdef np.double_t out = 0
    
    for i in range(n):
        out += (x[i] - mean) ** 2
    return out / (n - 1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline np.double_t sign(np.double_t x) nogil:
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int imin(int x, int y) nogil:
    if x < y:
        return x
    else: 
        return y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int imax(int x, int y) nogil:
    if x > y:
        return x
    else: 
        return y
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int rint(np.double_t x) nogil:
    return <int> round(x)

def calc_bandwidth(x, int nb=1000):
    x = np.asarray(x)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)

    cdef int n = len(x)
    cdef np.double_t x_min = x.min()
    cdef np.double_t x_max = x.max()
    cdef np.double_t[:] x_view = x
    cdef np.double_t* x_ptr = &x_view[0]
    cdef np.double_t ret = 0.0
    with nogil:
        ret = py_bcv(x_ptr, n, x_min, x_max, nb)
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline np.double_t py_bcv(np.double_t* x, int n, np.double_t x_min, np.double_t x_max, int nb=1000) nogil:
    cdef np.double_t h, lower, upper
    cdef density den
    if(n == 0):
        printf("'x' has length zero\n")
        return -1
    hmax = 1.144 * sqrt(r_var(x, n)) * (n**(-1.0/5)) * 4
    lower = 0.1 * hmax
    upper = hmax
    den = VR_den_bin(n, nb, x, x_min, x_max)
    
    # Optimize
    cdef np.double_t x1 = lower, x2 = upper, xatol = 0.1*lower
    cdef int maxfun = 500, num
    cdef np.double_t sqrt_eps = sqrt(2.2e-16), golden_mean = 0.5 * (3.0 - sqrt(5.0)), a, b, nfc, xf, fulc, 
    cdef np.double_t rat, e, fx, ffulc, fnfc, xm, tol1, tol2, r, p, q, golden, si, fu

    a = x1; b = x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    h = xf
    fx = VR_bcv_bin(n, nb, den, h)
    num = 1

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while (fabs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if fabs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = fabs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((fabs(p) < fabs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                h = xf + rat

                if ((h - a) < tol2) or ((b - h) < tol2):
                    si = sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden section step
                golden = 1

        if golden:  # Do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = sign(rat) + (rat == 0)
        h = xf + si * fmax(fabs(rat), tol1)
        fu = VR_bcv_bin(n, nb, den, h)
        num += 1

        if fu <= fx:
            if h >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = h, fu
        else:
            if h < xf:
                a = h
            else:
                b = h
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = h, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = h, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            break
    h = xf
    free(den.cnt)
    if(h < 1.1*lower or h > upper-0.1*lower):
        printf("py Warning: minimum occured at one end of the range\n")
    return h
