import numpy as np

def Poisson(lambda_, k):
    """Returns the value of the Poisson probability distribution for
        a given value of lambda and k.
        P_lambda(k) = lambda^k * exp(-lambda) / k!"""

    log_p = k*np.log(lambda_) - lambda_
    # Sum over -ln(n) from n=1 to n=k
    for i in range(k):
        log_p -= np.log(i+1)
        
    return np.exp(log_p)

# Array of (lambda, k) to compute
params_arr = np.array([[1, 0], [5, 10], [3, 21],
                       [2.6, 40], [100, 5], [101, 200]])


for params in params_arr:
    print(rf"P_{params[0]}({int(params[1])}) = {Poisson(np.float32(params[0]), np.int32(params[1])):.7e}")