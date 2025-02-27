# Based on the minimal routine provided in the exercise set at
# https://home.strw.leidenuniv.nl/~daalen/Handin_files/vandermonde.py

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import copy
import timeit

# Simple Crout's algorithm
def Crouts(A):
    """Performs simple Crouts algorithm to perform LU decomposition
    
    Arguments:
        A: nxn matrix
        
    Returns:
        2D list containing L and U matrix such that a=LU.
        U is contained in the upper-right half, including diagonal
        L is contained in the bottom-left half, excluding diagonal. The
        diagonal elements of L are 1 and are not returned."""
        
    a = copy.deepcopy(A)
    # Loop over the columns j
    N = len(a)
    for j in range(N):
        # For each j, beta_0j = a_0j. All beta_ij (i<=j) can be expressed
        # in previously calculated values/
        # The two sums below could be taken together, but splitting them
        # is computationally less expensive than deciding whether to divide
        # by beta_jj through if-statements at each iteration.
        
        # First sum over the beta. Define i such that it corresponds to index
        for i in range(1,j+1):
            for k in range(i):
                a[i][j] -= a[i][k]*a[k][j]
                
        # Then sum over the alpha
        for i in range(j+1, N):
            for k in range(j):
                a[i][j] -= a[i][k]*a[k][j]
            a[i][j] /= a[j][j] # Only at the end divide by beta_jj
            
    return a


def solve_system(a, y_vals):
    """Solve matrix system, ac = y, after performing LU decomposition
    
    Arguments:
        a: combined LU matrix, as output by the Crouts function.
        y_vals: y vector on RHS of system
        
    Returns:
        array containing c vector solving the system ac=y"""
    
    y = copy.deepcopy(y_vals)
    N=len(y)
    
    # Forward substitution
    for i in range(1,N):
        for j in range(i):
            y[i] -= a[i][j]*y[j]
            
    # Backward substitution
    # i = 0 corresponds to the last index: x[N-1]
    for i in range(N):
        for j in range(i):
            y[N-1-i] -= a[N-1-i][N-1-j]*y[N-1-j]
        y[N-1-i] /= a[N-1-i][N-1-i]
        
    return y


def iterative_Crouts_improvement(A, a, y_vals, c, N):
    """Iteratively improve on Crout's algorithm.
    
    Arguments:
        A, matrix: original nxn matrix
        a, matrix: combined LU matrix, as output by the Crouts function.
        c, array: of coefficients as output by solve_system function
        N, int: Number of iterations
        
    Returns:
        array containing improved c vector solving the system ac=y
        """
    
    cn = copy.deepcopy(c)
    
    for i in range(N):
        # Find difference between true y-values and those found from Crout's
        delta_y = A@cn - y_vals
        # Solve for delta_c
        delta_cn = solve_system(a, delta_y)
        cn -= delta_cn
    
    return cn


def get_y(x_interp, c):
    """Get y-values given x_interp and c corresponding to
    y = Sum(c_i * x_interp^i).
    
    Arguments:
        x_interp:   array with x-values at which to compute the y-value
        c:          array of coefficients as output by solve_system function
        
    Returns:
        array of y-values corresponding to given x-values."""
        
    x_powers = np.array([x_interp**i for i in range(len(c))])
    return c@x_powers


def LU_interpolation(x_interp, x_vals, y_vals, 
                     iterations=None, print_c = False):
    """Applies LU decomposition to interpolate.
    
    Arguments:
        x_interp, array:   x-values to interpolate at
        x_vals, array:     x-values of data
        y_vals, array:     y-values of data
        iterations, int (optional):
                        Number of iterations improvements on LU decomposition
                        to perform. Default = None.
        print_c, bool (optional):
                        If True, print the coefficients (after iteration)
                        in a file names coefficient.txt. Default = False
                        
    Returns
        array with interpolated y-values.
        """
    
    # Generate the matrix V
    V = np.ones([len(x), len(x)])
    for i in range(1, len(x)):
        V[:,i] = x**i
        
    #Perform LU decomposition and find the coefficients
    V_crout = Crouts(V)
    c = solve_system(V_crout, y)
    
    if iterations:
        c = iterative_Crouts_improvement(V, V_crout, y_vals, c, iterations)
        
    if store_c:
        for i in range(len(c)):
            print(f"c_{i} = {c[i]:.2e}")
        print()
        
    y_interp = get_y(x_interp, c)
        
    return y_interp


def Neville_interpolation(x_interp, x_vals, y_vals):
    """Applies Nevilles algorithm to interpolate.
    
    Arguments:
        x_interp, array:   x-values to interpolate at
        x_vals, array:     x-values of data
        y_vals, array:     y-values of data
                        
    Returns:
        array with interpolated y-values.
        """
    
    y_interp = np.tile(y_vals, (len(x_interp), 1))
    N = len(x_vals)

    # Loop repeatedly over all intervals
    for i in range(N-1):
        for j in range(N-1-i):
            y_interp[:,j] = ((x_vals[j+1+i]-x_interp)*y_interp[:,j]
                           + (x_interp-x_vals[j])*y_interp[:,j+1]) /(
                               x_vals[j+1+i]-x_vals[j])
            
    return y_interp[:,0]


# Load the data
data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]

# Generate interpolation x-values
xx=np.linspace(x[0],x[-1],1001) # x values to interpolate at

# Calculate interpolated y-values
yya = LU_interpolation(xx, x, y)
ya = LU_interpolation(x, x, y)

yyb = Neville_interpolation(xx, x, y)
yb = Neville_interpolation(x, x, y)

yyc1 = LU_interpolation(xx, x, y, iterations=1)
yc1 = LU_interpolation(x, x, y, iterations=1)
yyc10 = LU_interpolation(xx, x, y, iterations=10)
yc10 = LU_interpolation(x, x, y, iterations=10)


#Plot of points with absolute difference shown on a log scale (question 2a)
fig=plt.figure()
gs=fig.add_gridspec(2,hspace=0,height_ratios=[2.0,1.0])
axs=gs.subplots(sharex=True,sharey=False)
axs[0].plot(x,y,marker='o',linewidth=0)
plt.xlim(-1,101)
axs[0].set_ylim(-400,400)
axs[0].set_ylabel('$y$')
axs[1].set_ylim(1e-20,1e1)
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line,=axs[0].plot(xx,yya,color='orange')
line.set_label('Via LU decomposition')
axs[0].legend(frameon=False,loc="lower left")
axs[1].plot(x,abs(y-ya),color='orange')
plt.savefig('./plots/LU_interpolation.png',dpi=600)


#For questions 2b and 2c, add this block
line,=axs[0].plot(xx,yyb,linestyle='dashed',color='green')
line.set_label('Via Neville\'s algorithm')
axs[0].legend(frameon=False,loc="lower left")
axs[1].plot(x,abs(y-yb),linestyle='dashed',color='green')
plt.savefig('./plots/Neville_interpolation.png',dpi=600)

#For question 2c, add this block too
line,=axs[0].plot(xx,yyc1,linestyle='dotted',color='red')
line.set_label('LU with 1 iteration')
axs[1].plot(x,abs(y-yc1),linestyle='dotted',color='red')
line,=axs[0].plot(xx,yyc10,linestyle='dashdot',color='purple')
line.set_label('LU with 10 iterations')
axs[1].plot(x,abs(y-yc10),linestyle='dashdot',color='purple')
axs[0].legend(frameon=False,loc="lower left")
plt.savefig('./plots/LU_iterative.png',dpi=600)

# #Don't forget to caption your figures to describe them/
# #mention what conclusions you draw from them!


### Timeit analysis of computation times
# Argument needs to be a string or callable, so we can pass a dummy lambda
# function which computes the interpolations to timeit.timeit

execution_time_a=timeit.timeit(lambda: LU_interpolation(xx,x,y),number=100)
execution_time_b=timeit.timeit(lambda: Neville_interpolation(xx,x,y),
                               number=100)
execution_time_c=timeit.timeit(lambda: LU_interpolation(xx,x,y,iterations=10),
                               number=100)

# Test how long LU decomposition takes. Manually compute V, then only decompose
def get_V(x):
    """Returns the Vandermonde matrix given an array of x-values
    
    Arguments:
        x, array: array of x-values
        
    Returns:
        Vandermonde matrix"""
        
    V = np.ones([len(x), len(x)])
    for i in range(1, len(x)):
        V[:,i] = x**i
    return V
        
V = get_V(x)

execution_time_decomposition = timeit.timeit(lambda: Crouts(V), number=100)

print("Computation times for 1000 interpolated points, 100 repetitions:")
print(f'LU interpolation: t = {execution_time_a} ms')
print(f'Neville interpolation: t = {execution_time_b} ms')
print(f'LU interpolation with 10 iterations: t = {execution_time_c} ms')
print(f'LU decomposition only: t = {execution_time_decomposition} ms')


# Test what the times are for 10000 interpolated points.
xx10000 = xx=np.linspace(x[0],x[-1],10001)

execution_time_a10000=timeit.timeit(lambda: LU_interpolation(xx10000,x,y),
                                    number=100)
execution_time_b10000=timeit.timeit(lambda: Neville_interpolation(xx10000,x,y),
                                    number=100)
execution_time_c10000=timeit.timeit(lambda: LU_interpolation(xx10000,x,y,
                                                             iterations=10),
                                    number=100)

print("Computation times for 10000 interpolated points, 100 repetitions:")
print(f'LU interpolation: t = {execution_time_a10000} ms')
print(f'Neville interpolation: t = {execution_time_b10000} ms')
print(f'LU interpolation with 10 iterations: t = {execution_time_c10000} ms')
