
import numpy as np
import insarsyn


def test_gen_noisy_stack():
    stack_dims = (3, 4, 5)
    amps = np.ones(stack_dims)
    phis = np.zeros(stack_dims)
    cohs = 0.7*np.ones((stack_dims[0], *stack_dims))

    assert stack_dims == insarsyn.gen_noisy_stack(amps, phis, cohs).shape
