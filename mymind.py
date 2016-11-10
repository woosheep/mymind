#coding:utf8
#woosheep@20160518

import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

#derivation of 	sigmoid
def dsigmoid(x):
	return x*(1-x)
	
#update weight
def updw(w, l, delta):
	#dw=delta*l
	return w+ l.T*delta

#forward propagation
def fp(l, w):
	return sigmoid(np.dot(l, w))

#back propagation
def bp(err, l, w=[]):
	#delta=err*df/sum(delta*w)*df
	return err* dsigmoid(l) if w == [] else err.dot(w.T).sum() *dsigmoid(l)

def loss(t, o):
	return np.mean(np.abs(t[0]-o[0]))

def matrix(x):
	return np.array([x])

def trainn(x, y, w):
	num = len(w)
	delta=[0]*num
	l=[None]*(num+1)
	l[0] =x
	while True:
		'''
	    l1 = fp(l0,w0)
	    l2 = fp(l1,w1)
	    l3 = fp(l2,w2)
	    '''
		for i in range(1, num+1):
			l[i]=fp(l[i-1],w[i-1])

		'''
		if loss(y, l3) < 0.01:
			return w0, w1, w2
		'''
		if loss(y, l[num]) < 0.01:
			return w
		'''
	    l2_delta = bp((y-l3),l3)
	    w2 = updw(w2, l2, l2_delta) 

	    l1_delta = bp(l2_delta, l2, w2)
	    w1  = updw(w1, l1, l1_delta) 

	    l0_delta = bp(l1_delta, l1, w1)
	    w0  = updw(w0, l0, l0_delta)
	    '''
		for i in range(num, 0,-1):
			if i == num:
				##y-l[i]is[[]],val is [0][0]
				delta[i-1] =bp(((y-l[i])[0][0]),l[i])
				w[i-1] = updw(w[i-1],l[i-1],delta[i-1])
			else:
				delta[i-1] =bp(delta[i],l[i],w[i])
				w[i-1] = updw(w[i-1],l[i-1],delta[i-1])

		
def train(x, y, deep=11, node=9):
	row_num = x.shape[1]
	w = []
	for i in range(deep):
		if i == 0:
			w.append(np.random.random((x.shape[1],node))) 
		elif i == deep -1:
			w.append(np.random.random((node, y.shape[1]))) 
		else:
			w.append(np.random.random((node,node)))
	for i in range(x.shape[0]):
		w=trainn(matrix(x[i]), matrix(y[i]), w)
	return w
	'''	
		w0 = np.random.random((x.shape[1],node)) 
		w1 = np.random.random((node,node)) 
		w2 = np.random.random((node, y.shape[1])) 
	for i in range(x.shape[0]):
		w0, w1,w2=trainn(matrix(x[i]), matrix(y[i]), w0, w1, w2)
	return w0, w1, w2
	'''


def predict(x, w):
	val =0
	for i in range(len(w)):
		if i ==0:
			val=fp(x,w[i])
		else:
			val=fp(val, w[i])
	return val


x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1],
                [0,0]])
y = np.array([[0,0,1,1,0]]).T  
 



'''
deep =36
w=train(x, y, deep)
print predict([1,1],w)[0]
print predict([0,0],w)[0]
print predict([0,1],w)[0]
print predict([1,0],w)[0]
'''
for i in range(3,100):
	deep =11
	w=train(x, y, deep,i)
	print 'node:'+str(i)
	print predict([1,1],w)[0]
	print predict([0,0],w)[0]
	print predict([0,1],w)[0]
	print predict([1,0],w)[0]