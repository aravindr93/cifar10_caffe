'''
Python script to train a CNN on CIFAR-10 data set
	23rd May 2016
	Aravind Rajeswaran, IIT Madras
'''

# Setup + Data 
#=======================================================
import numpy as np
import scipy as sp
from pylab import *
import matplotlib.pyplot as plt

caffe_root = '/home/aravind/deep_learning/caffe/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe

import os
os.chdir(caffe_root)

print "**** CIFAR-10 data has already been downloaded and prepared"
#=======================================================
#
#
#
#
# Network Architecture
#=======================================================
from caffe import layers as L, params as P

def mynet(lmdb, batch_size):
    # 4 conv layers, each followed by a max_pool; followed by ReLU; and then 2 fully connected layers
    n = caffe.NetSpec()
    
    # Data Layer ========================================
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(mean_file="examples/cifar10/mean.binaryproto"), ntop=2)
    
    # 1st set Conv ======================================
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=128, pad=2, stride=1, 
                            weight_filler=dict(type='gaussian', std=0.0001), 
                            bias_filler=dict(type='constant', value=0))
    n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)
    
    # 2nd set Conv ======================================
    n.conv2 = L.Convolution(n.relu1, kernel_size=5, num_output=64, pad=2, stride=1,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    
    # 3rd set Conv ======================================
    n.conv3 = L.Convolution(n.pool2, kernel_size=5, num_output=32, pad=2, stride=1,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.pool3 = L.Pooling(n.relu3, kernel_size=3, stride=2, pool=P.Pooling.AVE)

    # 4th set Conv ======================================
    n.conv4 = L.Convolution(n.pool3, kernel_size=5, num_output=32, pad=2, stride=1,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.pool4 = L.Pooling(n.relu4, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    
    # 1st set FC ========================================
    n.fc1 =   L.InnerProduct(n.pool4, num_output=128,
                             weight_filler=dict(type='gaussian', std=0.1),
                             bias_filler=dict(type='constant', value=0))
    n.relu5 = L.ReLU(n.fc1, in_place=True)
    
    # 2nd set FC ========================================
    n.fc2 =   L.InnerProduct(n.relu5, num_output=128,
                             weight_filler=dict(type='gaussian', std=0.1),
                             bias_filler=dict(type='constant', value=0))
    n.relu6 = L.ReLU(n.fc2, in_place=True)
    
    # Score and output ==================================
    n.score = L.InnerProduct(n.relu6, num_output=10,
                             weight_filler=dict(type='gaussian', std=0.1),
                             bias_filler=dict(type='constant', value=0))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()

train_net_path = 'examples/cifar10/mynet/mynet_auto_train.prototxt'
test_net_path = 'examples/cifar10/mynet/mynet_auto_test.prototxt'
solver_config_path = 'examples/cifar10/mynet/mynet_auto_solver.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(mynet('examples/cifar10/cifar10_train_lmdb', 128)))    
with open(test_net_path, 'w') as f:
    f.write(str(mynet('examples/cifar10/cifar10_test_lmdb', 100)))
#=======================================================
#
#
#
#
# Solver Options
#=======================================================
caffe.set_mode_cpu()
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

s.random_seed = 0xCAFFE

# Specify locations of the train and (maybe) test networks.
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 2500 		# Test after every 2500 training iterations.
s.test_iter.append(100) 	# Test on 100 batches each time we test.
s.max_iter = 10000     		# no. of times to update the net (training iterations)

s.type = "Nesterov"
s.base_lr = 0.01  
s.momentum = 0.9
s.weight_decay = 0.004

# Set `lr_policy` to define how the learning rate changes during training.
#s.lr_policy = "fixed"
s.lr_policy = 'inv'
s.gamma = 0.0001
s.power = 0.75

# Display and snapshot after 'x' iterations
s.display = 100
s.snapshot = 2500
s.snapshot_prefix = 'examples/cifar10/mynet/mynet'

# set solver mode to CPU
s.solver_mode = caffe_pb2.SolverParameter.GPU

# Write the solver to a temporary file and return its filename.
with open(solver_config_path, 'w') as f:
    f.write(str(s))
#=======================================================
#
#
#
#
# Training the network
#=======================================================

solver = None 
solver = caffe.get_solver(solver_config_path)

niter = 200  # EDIT HERE increase to train for longer
test_interval = 20

# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
test_loss = zeros(int(np.ceil(niter / test_interval)))

print "**** Entering Solver Loop ****"
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
      
    # Display results once in a while
    if it % test_interval == 0:
        correct = 0
		t_loss = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
			t_loss += solver.test_nets[0].blobs['loss'].data	    
        test_acc[it // test_interval] = correct/1e4
		test_loss[it // test_interval] = t_loss/100
        print '**** Iteration', it, '| train loss = ', train_loss[it], '| test loss = ', test_loss[it // test_interval], ' | accuracy = ', test_acc[it // test_interval]
#=======================================================
#
#
#
#
# Plot
#=======================================================
fig1 = plt.figure(figsize=(10,5))
ax = fig1.add_subplot(111)
ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.plot(arange(niter), train_loss, 'b',
ax.plot(test_interval * arange(len(test_loss)), test_loss, 'r')
ax.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.savefig('learning_curve_mynet')

fig2 = plt.figure(figsize=(10,5))
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration', fontsize=14, fontweight='bold')
ax1.set_ylabel('train loss', fontsize=14, fontweight='bold')
ax2.set_ylabel('test accuracy', fontsize=14, fontweight='bold')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]), fontsize=14, fontweight='bold')
plt.savefig('test_accuracy_mynet.png') 




