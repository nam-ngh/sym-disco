import numpy as np

def ermakov_pinney(pts):
    '''
    Prolonged sl(2,R) generators for u'' = u^-3, evaluated on the data.
    pts: (N, 4) columns [x, u, u', u'']
    returns: (3, N, 4) -- [X1, X2, X3] x points x (xi, eta, eta1, eta2) ie 3 vector fields as coordinates
    '''
    x, u, du, ddu = pts.T

    # X1 = d_x
    X1 = np.stack([
        np.ones_like(x), np.zeros_like(x),
        np.zeros_like(x), np.zeros_like(x)
    ], axis=-1)

    # X2 = 2x d_x + u d_u
    X2 = np.stack([2.0 * x, u, -du, -3.0 * ddu], axis=-1)

    # X3 = x^2 d_x + xu d_u
    X3 = np.stack([x**2, x * u, u - x * du, -3.0 * x * ddu], axis=-1)

    return np.stack([X1, X2, X3])