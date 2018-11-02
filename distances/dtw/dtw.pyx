import numpy as np 
cimport numpy as np 
np.import_array()

from libc.float cimport DBL_MAX

cdef inline double min_c(double a, double b): return a if a <= b else b
cdef inline int max_c_int(int a, int b): return a if a >= b else b
cdef inline int min_c_int(int a, int b): return a if a <= b else b

# it takes as argument two time series with shape (l,m) where l is the length
# of the time series and m is the number of dimensions 
# for multivariate time series
# even if we have univariate time series, we should have a shape equal to (l,1)
# the w argument corrsponds to the length of the warping window in percentage of 
# the smallest length of the time series min(x,y) - if negative then no warping window
# this funuction assumes that x is shorter than y 
def dynamic_time_warping(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y , w = -1):
    # make sure x is shorter than y 
    # if not permute
    cdef np.ndarray[double, ndim=2] X = x 
    cdef np.ndarray[double, ndim=2] Y = y 
    cdef np.ndarray[double, ndim=2] t
    
    if len(X)>len(Y): 
        t = X
        X = Y 
        Y = t
        
    cdef int r,c, im,jm, i, j, lx, jstart, jstop, idx_inf_left, ly  
    cdef double curr 
    
    lx = len(X)
    ly = len(Y)
    r = lx + 1
    c = ly +1 
    if w < 0: 
        w = max_c_int(lx,ly)
    else: 
        w = w*max_c_int(lx,ly)
    
    cdef np.ndarray[double, ndim=2] D = np.zeros((r,c),dtype=np.float64)
    
    D[0,1:] = DBL_MAX
    D[1:,0] = DBL_MAX 
    
    D[1:,1:] = np.square(X[:,np.newaxis]-Y).sum(axis=2).astype(np.float64)
    
    for i in range(1,r):
        jstart = max_c_int(1 , i-w)
        jstop = min_c_int(c , i+w+1)
        idx_inf_left = i-w-1
        
        if idx_inf_left >= 0 :
            D[i,idx_inf_left] = DBL_MAX
            
        for j in range(jstart,jstop):
            im = i-1
            jm = j-1
            D[i,j] = D[i,j] + min_c(min_c(D[im,j],D[i,jm]),D[im,jm])
            
        if jstop < c:
            D[i][jstop] = DBL_MAX
    
    return np.sqrt(D[lx,ly]),D

