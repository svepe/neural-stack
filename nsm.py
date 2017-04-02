import numpy as np
import chainer.functions as F
from chainer import Variable


def neural_stack(V, s, d, u, v):
    # strengths
    s_new = d
    for t in reversed(xrange(s.shape[1])):
        x = s[:, t].reshape(-1, 1) - u
        s_new = F.concat((s_new, F.maximum(Variable(np.zeros_like(x.data)), x)))
        u = F.maximum(Variable(np.zeros_like(x.data)), -x)

    s = F.fliplr(s_new)

    # memory
    V = F.concat((V, F.expand_dims(v, 1)))

    # result
    r = Variable(np.zeros_like(v.data))
    ur = Variable(np.ones_like(u.data))
    for t in reversed(xrange(s_new.shape[1])):
        w = F.minimum(s[:, t].reshape(-1, 1), ur)
        r += V[:, t] * F.broadcast_to(w, V[:, t].shape)
        x = ur - s[:, t].reshape(-1, 1)
        ur = F.maximum(Variable(np.zeros_like(x.data)), x)

    return V, s, r

batch_size = 3
stack_element_size = 2

V = Variable(np.zeros((batch_size, 1, stack_element_size)))
s = Variable(np.zeros((batch_size, 1)))

d = Variable(np.ones((batch_size, 1)) * 0.4)
u = Variable(np.ones((batch_size, 1)) * 0.)
v = Variable(np.ones((batch_size, stack_element_size)))
V, s, r = neural_stack(V, s, d, u, v)

d = Variable(np.ones((batch_size, 1)) * 0.8)
u = Variable(np.ones((batch_size, 1)) * 0.)
v = Variable(np.ones((batch_size, stack_element_size)) * 2.)
V, s, r = neural_stack(V, s, d, u, v)

d = Variable(np.ones((batch_size, 1)) * 0.9)
u = Variable(np.ones((batch_size, 1)) * 0.9)
v = Variable(np.ones((batch_size, stack_element_size)) * 3.)
V, s, r = neural_stack(V, s, d, u, v)

d = Variable(np.ones((batch_size, 1)) * 0.1)
u = Variable(np.ones((batch_size, 1)) * 0.1)
v = Variable(np.ones((batch_size, stack_element_size)) * 3.)
V, s, r = neural_stack(V, s, d, u, v)

print V.data
print s.data
print r.data
