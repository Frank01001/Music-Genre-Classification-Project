import jax.numpy as jnp
import numpy as np

N_utils = 0
in_labels = None
in_data = None


# The hessian is always equal to zero because the constraint is linear ALTRA CLASSE
def hessian(x):
    return np.zeros((N_utils, N_utils))


# This is the objective function of the dual problem with the usage of the kernel trick ALTRA CLASSE
def obj_kernel(c):
    partial_1 = -jnp.sum(c)

    c_outer = jnp.outer(c, c)
    c_outer = jnp.triu(c_outer)

    y_outer = jnp.outer(in_labels, in_labels)
    y_outer = jnp.triu(y_outer)

    K = (jnp.dot(in_data, in_data.T) + np.ones((N_utils, N_utils))) ** 3

    partial_2 = 0.5 * jnp.sum(c_outer * y_outer * K)

    return partial_1 + partial_2
