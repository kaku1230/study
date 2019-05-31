import numpy as np
import matplotlib.pyplot as plt
X=2*np.random.rand(100,1)
y=4+3*X+np.random.randn(100,1)
X_b=np.c_[np.ones((100,1)),X]
e_epochs=500
t0,t1=5,50
m=100
def learning_schedule(t):
    return t0/(t+t1)

theta=np.random.randn(2,1)
for epoch in range(e_epochs):
    for i in range(m):
        random_index=np.random.randint(m)
        xi=X_b[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients=xi.T.dot(xi.dot(theta)-yi)
        learning_rate=learning_schedule(epoch*m+i)
        theta=theta-learning_rate*gradients

print(theta)
print(gradients)
