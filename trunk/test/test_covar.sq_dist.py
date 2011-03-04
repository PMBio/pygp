import numpy as NP
import sq_dist_new as D_N
import sq_dist as D

x1 = NP.linspace(0,1,10000).reshape(-1,1)

def sq1(): return D.dist(x1,x1)
def sq2(): return D_N.dist(x1,x1)

from timeit import Timer

t1 = Timer(sq1)
t2 = Timer(sq2)

print "old dist: %f.15" % (t1.timeit(number=1))
print "new dist: %f.15" % (t2.timeit(number=1))

D_d = D.dist(x1,x1)
D_N_d = D_N.dist(x1,x1)
assert (D_d==D_N_d).all(), "Not right format"
