
import numpy
import pylab

trees = 1000
directory = '/home/singhala/rf_parallel/test_car/output/'

err_strings = [open(directory + 'MSE_%d.txt' % i).readline().split() \
               for i in xrange(trees)]
err_nums = [[float(s) for s in errs] for errs in err_strings]
MSEs = numpy.array([sum(errs)/len(errs) for errs in err_nums])
x = numpy.arange(0, trees, 1)
pylab.plot(x, MSEs)
pylab.show()
