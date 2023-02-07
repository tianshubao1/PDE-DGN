# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:50 2019

@author: tianshubao
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
import math

''' Declare constants '''
learning_rate = 0.01
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 200    #70
batch_size = 2000       #ignore this
hidden_size = 20 
input_size = 20-2
phy_size = 2        #physical variable
T = 13149               #13149 time points,36 years
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
npic = 12       #time series 
n_steps = int(4383/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1        #sliding window, so more sequences
N_seg = 42                  # number of rivers
kb=1.0                      # dropout, no dropout

''' Build Graph '''

tf.reset_default_graph()
random.seed(9001)
# Graph input/output
x = tf.placeholder("float", [None, n_steps,input_size]) #tf.float32 # none is flexible   #( , 365, 20)
p = tf.placeholder("float", [None, n_steps,phy_size])
y = tf.placeholder("float", [None, n_steps]) #tf.int32
m = tf.placeholder("float", [None, n_steps])
#A = tf.placeholder("float", [N_seg,N_seg])


A_dist_up = tf.placeholder("float", [42, 42])
A_dist_dn = tf.placeholder("float", [42, 1]) #only 1 down stream
A_upindex = tf.placeholder("float", [42, 42])
A_dnindex = tf.placeholder("float", [42, 42])
sflow = tf.placeholder("float", [None, n_steps])

keep_prob = tf.placeholder(tf.float32)

carea = tf.get_variable('carea',[N_seg], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
DL = tf.get_variable('DL',[N_seg], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))


W2 = tf.get_variable('W_2',[hidden_size, n_classes], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b2 = tf.get_variable('b_2',[n_classes],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

Wg1 = tf.get_variable('W_g1',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bg1 = tf.get_variable('b_g1',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))
Wg2 = tf.get_variable('W_g2',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bg2 = tf.get_variable('b_g2',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))


Wa1 = tf.get_variable('W_a1',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
Wa2 = tf.get_variable('W_a2',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
ba = tf.get_variable('b_a',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

Wc1 = tf.get_variable('W_c1',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
Wc2 = tf.get_variable('W_c2',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bc = tf.get_variable('b_c',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))


Wp = tf.get_variable('W_p',[hidden_size, phy_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bp = tf.get_variable('b_p',[phy_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))



lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0) 
#tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0) 
#lstm_cell = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]


o_sr = []
#o_hr = []
#o_gr = []
o_phy = []

#output_series, current_state = tf.nn.dynamic_rnn(lstm_cell, tf.expand_dims(x[:,0,:],axis=1), dtype=tf.float32) #46*(500*7)
#output_series, current_state = lstm_cell(x[:,0,:], lstm_cell.get_initial_state(inputs=x[:,0,:])) #46*(500*7)
output_series, current_state = lstm_cell(x[:,0,:], lstm_cell.zero_state(N_seg, dtype=tf.float32)) #46*(500*7)




c_pre,h_pre = current_state
state_graph = tf.contrib.rnn.LSTMStateTuple(c_pre,h_pre)
#h_gr = tf.matmul(A,tf.matmul(h,Wg))
## whether transform the state as well
#c_gr = tf.matmul(A,tf.matmul(c,Wg))
#state_graph = current_state #tf.contrib.rnn.LSTMStateTuple(c_gr,h_gr)
o_sr.append(h_pre)
#o_hr.append(h_pre)
#o_gr.append(h_pre)
#o_sr.append(h_gr)
o_phy.append(tf.matmul(h_pre,Wp)+bp)


identity = tf.eye(42)        

As = []
Us = []

#with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
    
    
for t in range(1,n_steps):  #river other segments    n_steps = 365
    
    A = []
    deltat = 1
    U = sflow[:, t]/ carea  #42*1 vector, broadcast

    up_min = A_dist_up  #42* 42 matrix
    dn_min = A_dist_dn  #convert to 42*42 matrix, place the values at corresponding locations
        

    coef1 = 2*deltat*DL/(up_min*(up_min + dn_min))    #  dn_min is a column vector

        
    coef2 = - 2*deltat*DL*( 1/(dn_min*(up_min + dn_min)) + 1/(up_min*(up_min + dn_min)) )   #values are at up_min locations
    coef2 = tf.reduce_sum(coef2, 1) #becomes a column vector
    coef2 += U*deltat/dn_min
        # coef2[coef2 == np.inf] = 0 #remove non-reachable points
        # coef2[coef2 == -np.inf] = 0
        # coef2.sum(axis=1)
        
    coef3 = 2*deltat*DL/(dn_min*(up_min + dn_min))
    coef3 = tf.reduce_sum(coef3, 1)
    coef3 += -U*deltat/dn_min
    
        # coef3[coef3 == np.inf] = 0 #remove non-reachable points
        # coef3[coef3 == -np.inf] = 0
        # coef3.sum(axis=1)
        
    A = coef1 * A_upindex + coef2 * identity + coef3 * A_dnindex    
    # A = tf.matmul(A, tf.matmul(A, A))
    # A = tf.matmul(A, A) #A^4
           
    output_series, current_state = lstm_cell(x[:,t,:], state_graph) #46*(500*7)
    # run one time
    c,h = current_state  #contains 2 c and h
    h_gr = tf.nn.tanh(tf.matmul(A,tf.matmul(h_pre,Wg1)+bg1))        #（4） h_gr is q, transfer equation #A is adjacent matrix
    # whether transform the state as well
    c_gr = tf.nn.tanh(tf.matmul(A,tf.matmul(c_pre,Wg2)+bg2))        #4

    
    h_pre = tf.nn.sigmoid(tf.matmul(h,Wa1)+tf.matmul(h_gr,Wa2)+ba)   #h_curr   #2
    c_pre = tf.nn.sigmoid(tf.matmul(c,Wc1)+tf.matmul(c_gr,Wc2)+bc)   #c_curr   #2
    state_graph = tf.contrib.rnn.LSTMStateTuple(c_pre,h_pre)
    o_sr.append(h_pre)      #all out put

    As.append(A)
    Us.append(U)

weights = tf.stack(As, axis=2)
o_sr = tf.stack(o_sr,axis=1) # N_seg - T - hidden_size
oh = tf.reshape(o_sr,[-1,hidden_size])



oh = tf.nn.dropout(oh,keep_prob)

y_prd = tf.matmul(oh,W2)+b2

#  cost function
y_prd_fin = tf.reshape(y_prd,[-1,n_steps])
y_prd = tf.reshape(y_prd,[-1])
cost_sup = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((y_prd-tf.reshape(y,[-1])),tf.reshape(m,[-1]))))/
                   (tf.reduce_sum(tf.reshape(m,[-1]))+1))


tvars = tf.trainable_variables()
for i in tvars:
    print(i)
saver = tf.train.Saver(max_to_keep=3)








##this part is pre train
#optimizer_pre = tf.train.AdamOptimizer(learning_rate_pre)
##grads = tf.gradients(cost, tvars)
#gvs_pre = optimizer_pre.compute_gradients(cost_phy)
##capped_gvs_pre = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_pre]
#train_op_pre = optimizer_pre.apply_gradients(gvs_pre)





#this part is train
optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(cost_sup)
#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(gvs)
#train_op = optimizer.apply_gradients(zip(grads, tvars))




''' Load data '''
feat = np.load('processed_features.npy')
label = np.load('sim_temp.npy') #np.load('obs_temp.npy')
obs = np.load('obs_temp.npy') #np.load('obs_temp.npy')
mask = (label!=-11).astype(int)
maso = (obs!=-11).astype(int)

seg_test = np.load('sel_test_id.npy')

#seg_test = np.load('sel_seg_hard.npy')
flow = np.load('sim_flow.npy')



feat = np.delete(feat,[9,10],2)
#read up, down and etc
#adj = np.load('up_dist.npy') #
adj_up = np.load('up_dist.npy')     #upstream
adj_dn = np.load('dn_dist.npy')


A_dist_up_hat = adj_up.copy()
A_dist_down_hat = adj_dn.copy()
# print(A_dist_down_hat.shape)
A_upindex_hat = np.zeros([N_seg, N_seg])
A_dnindex_hat = np.zeros([N_seg, N_seg])
flow_hat = []

for i in range(0,42):   #every row
    # A_dist_row_up = []
    # A_dist_row_down = []
    # A_upindex_row = []
    # A_dnindex_row = []
    # up_set = {}
    
    #direction = 0
    temp1 = adj_up[i].copy()
    # temp1 = np.delete(temp1, i)    # i is index position, no delete is needed
    # min1 = temp1[np.nonzero(temp1)]  #min1 is empty, all distances
    
    # if min1.size > 0:   #has upper stream
        # up_min = np.min(min1)
    count = 0    
    for k in range(0,42):
        if temp1[k] > 0:
            A_upindex_hat[i][k] = 1
            count += 1
            # up_set.add(up_index)            
            
    if count == 0:       #no upper stream
        A_dist_up_hat[i][i] = 10000 #float('inf')
        up_index = i
        A_upindex_hat[i][up_index] = 1
    
       
    temp2 = adj_dn[i].copy()
    temp2 = np.delete(temp2, i);
    min2 = temp2[np.nonzero(temp2)]  
    
    if min2.size > 0:
        dn_min = np.min(min2)
        dn_index = np.where(adj_dn[i] == dn_min)   
        A_dnindex_hat[i][dn_index] = 1
        
    else:   #no lower stream
        A_dist_down_hat[i][i] = 10000 #float('inf')
        dn_index = i   
        A_dnindex_hat[i][dn_index] = 1
        
    
A_dist_up_hat[A_dist_up_hat == 0] = 1800000
A_dist_down_hat = np.sum(A_dist_down_hat, 1)
A_dist_down_hat = np.reshape(A_dist_down_hat, (42,1))

        




x_te = feat[:,cv_idx*4383:(cv_idx+1)*4383,:] #12 years, 365 days, cv = cross validation, test
y_te = label[:,cv_idx*4383:(cv_idx+1)*4383]
o_te = obs[:,cv_idx*4383:(cv_idx+1)*4383]
m_te = mask[:,cv_idx*4383:(cv_idx+1)*4383]      #simulation mask
mo_te = maso[:,cv_idx*4383:(cv_idx+1)*4383]         #observation mask 
flow_te = flow[:,cv_idx*4383:(cv_idx+1)*4383] 


# temp = np.reshape(maso,[-1])
# # m_tr_2 = np.reshape(m_tr_2,[-1])
# s_perc = 0.1
# indices = np.random.choice(np.arange(42*13149), 
#                            replace=False, size=int(42*13149 * (1-s_perc)))
# temp[indices] = 0.0
# maso = temp.reshape((42, 13149))



if cv_idx==1:
    x_tr_1 = feat[:,:4383,:]    # one 2d plane
    # y_tr_1 = label[:,:4383]
    o_tr_1 = obs[:,:4383]
    # m_tr_1 = mask[:,:4383]
    mo_tr_1 = maso[:,:4383]
#    p_tr_1 = phy[:,:4383,:]
    flow_tr_1 = flow[:,:4383]
    
    x_tr_2 = feat[:,2*4383:3*4383,:]
    # y_tr_2 = label[:,2*4383:3*4383:]
    o_tr_2 = obs[:,2*4383:3*4383:]
    # m_tr_2 = mask[:,2*4383:3*4383:]
    mo_tr_2 = maso[:,2*4383:3*4383:] 
#    p_tr_2 = phy[:,2*4383:3*4383,:]
    flow_tr_2 = flow[:,2*4383:3*4383:]
    
    
if cv_idx==2:
    x_tr_1 = feat[:,:4383,:]
    # y_tr_1 = label[:,:4383]
    o_tr_1 = obs[:,:4383]
    # m_tr_1 = mask[:,:4383]
    mo_tr_1 = maso[:,:4383]
#    p_tr_1 = phy[:,:4383,:]
    flow_tr_1 = flow[:,:4383]
    
    
    x_tr_2 = feat[:,4383:2*4383,:]
    # y_tr_2 = label[:,4383:2*4383:]
    o_tr_2 = obs[:,4383:2*4383:]
    # m_tr_2 = mask[:,4383:2*4383:]
    mo_tr_2 = maso[:,4383:2*4383:]
#    p_tr_2 = phy[:,4383:2*4383,:]
    flow_tr_2 = flow[:,4383:2*4383:]



# mo_tr_1[9, :] = 0
# mo_tr_2[9, :] = 0


s_perc = 0.05 #0.2#0.4 #0.6 #0.8
#ts = np.sum(m_tr_1[seg_test,:])+np.sum(m_tr_2[seg_test,:])

indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
mo_tr_1[:,indices] = 0.0
indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
mo_tr_2[:,indices] = 0.0




# ### data on cold seasons
# for i in range(0, 365*12):        
#     if math.floor((i - 243)/365) > math.floor((i - 334)/365) :
#         mo_tr_1[:, i] = 0.0

# for i in range(0, 365*12):        
#     if math.floor((i - 243)/365) > math.floor((i - 334)/365) :
#         mo_tr_2[:, i] = 0.0

# for i in range(0, 365*12):        
#     if math.floor((i - 243)/365) == math.floor((i - 334)/365) :
#         mo_te[:, i] = 0.0  
        
        
        

x_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size])          # n_steps = 365
y_train_1 = np.zeros([N_seg*N_sec,n_steps])
o_train_1 = np.zeros([N_seg*N_sec,n_steps])
m_train_1 = np.zeros([N_seg*N_sec,n_steps])
mo_train_1 = np.zeros([N_seg*N_sec,n_steps])
#p_train_1 = np.zeros([N_seg*N_sec,n_steps,phy_size])
flow_train_1 = np.zeros([N_seg*N_sec,n_steps])

x_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size])
y_train_2 = np.zeros([N_seg*N_sec,n_steps])
o_train_2 = np.zeros([N_seg*N_sec,n_steps])
m_train_2 = np.zeros([N_seg*N_sec,n_steps])
mo_train_2 = np.zeros([N_seg*N_sec,n_steps])
#p_train_2 = np.zeros([N_seg*N_sec,n_steps,phy_size])
flow_train_2 = np.zeros([N_seg*N_sec,n_steps])


x_test = np.zeros([N_seg*N_sec,n_steps,input_size])
y_test = np.zeros([N_seg*N_sec,n_steps])
o_test = np.zeros([N_seg*N_sec,n_steps])
m_test = np.zeros([N_seg*N_sec,n_steps])
mo_test = np.zeros([N_seg*N_sec,n_steps])
p_test = np.zeros([N_seg*N_sec,n_steps,phy_size])
flow_test = np.zeros([N_seg*N_sec,n_steps])


for i in range(1,N_sec+1): #0 to 22
    x_train_1[(i-1)*N_seg:i*N_seg,:,:]=x_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    # y_train_1[(i-1)*N_seg:i*N_seg,:]=y_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_train_1[(i-1)*N_seg:i*N_seg,:]=o_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    # m_train_1[(i-1)*N_seg:i*N_seg,:]=m_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_1[(i-1)*N_seg:i*N_seg,:]=mo_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
#    p_train_1[(i-1)*N_seg:i*N_seg,:,:]=p_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    flow_train_1[(i-1)*N_seg:i*N_seg,:]=flow_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    
    x_train_2[(i-1)*N_seg:i*N_seg,:,:]=x_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    # y_train_2[(i-1)*N_seg:i*N_seg,:]=y_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_train_2[(i-1)*N_seg:i*N_seg,:]=o_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    # m_train_2[(i-1)*N_seg:i*N_seg,:]=m_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_2[(i-1)*N_seg:i*N_seg,:]=mo_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
#    p_train_2[(i-1)*N_seg:i*N_seg,:,:]=p_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    flow_train_2[(i-1)*N_seg:i*N_seg,:]=flow_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    
    x_test[(i-1)*N_seg:i*N_seg,:,:]=x_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_test[(i-1)*N_seg:i*N_seg,:]=y_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_test[(i-1)*N_seg:i*N_seg,:]=o_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_test[(i-1)*N_seg:i*N_seg,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_test[(i-1)*N_seg:i*N_seg,:]=mo_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    # mo_test_seg[(i-1)*N_seg:i*N_seg,:]=mo_te_seg[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
#    p_test[(i-1)*N_seg:i*N_seg,:,:]=p_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    flow_test[(i-1)*N_seg:i*N_seg,:]=flow_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    
    
    
    
    
''' Session starts '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())






print('Fine-tuning starts')
print('==================================')
#total_batch = int(np.floor(N_tr/batch_size))
los = 0
mre = 10
pred = np.zeros(y_test.shape)
weight_A = np.zeros([N_seg, N_seg, 4383])   #42,42, 12*365
d_l = np.zeros([N_seg, 4383])   #42,42, 12*365
Uss = np.zeros([N_seg, 4383]) 
# weight_A = np.zeros([N_seg, N_seg, N_sec])  #42,42, 12*365

for epoch in range(epochs): #range(epochs):
    if np.isnan(los):
            break
    alos = 0
    alos_s = 0
    alos_p = 0
    
    idx = range(N_sec)
    idx = random.sample(idx,N_sec)
    
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_1[index,:,:]
        batch_y = o_train_1[index,:]
        batch_m = mo_train_1[index,:]
#        batch_p = p_train_1[index,:,:]
        batch_f = flow_train_1[index,:]
        
#        # debug
#        prd,cc,hh,cg,hg,oo,oh,og = sess.run(
#            [y_prd,c,h,c_gr,h_gr,o_sr,o_hr,o_gr],
#            feed_dict = {
#                    x: batch_x,
#                    y: batch_y,
#                    m: batch_m,
#                    keep_prob: kb,
#                    A: A_hat
#        })
#        
        _,los_s = sess.run(
            [train_op, cost_sup],          #input different
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A_dist_up: A_dist_up_hat,
                    A_dist_dn: A_dist_down_hat,
                    A_upindex: A_upindex_hat,  
                    A_dnindex: A_dnindex_hat,
                    sflow: batch_f                     
                    
        })
        alos += los
        alos_s += los_s
#        alos_p += los_p
        if np.isnan(los):
            break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec))
    
    if np.isnan(los):
        break
        
    alos = 0
    alos_s = 0
    alos_p = 0
    
    
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_2[index,:,:]
        batch_y = o_train_2[index,:]
        batch_m = mo_train_2[index,:]
#        batch_p = p_train_2[index,:,:]
        batch_f = flow_train_2[index,:]

        _,los_s, wghs, dl = sess.run(
            [train_op, cost_sup, weights, DL],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A_dist_up: A_dist_up_hat,
                    A_dist_dn: A_dist_down_hat,
                    A_upindex: A_upindex_hat,  
                    A_dnindex: A_dnindex_hat,
                    sflow: batch_f                     
        })
        alos += los
        alos_s += los_s
        
        print(dl[6])
        
        # weight_9 = weight_A[8, 9, :]
        # np.save('flow_tr_2.npy', flow_tr_2)   
        
        # if i == 0:
        #     weight_A[:,:,:364] = wghs
        # else :            
        #     weight_A[:,:,int(364 + 364/2*(i - 1)): int(364 + 364/2*i) ] = wghs[:, :, int(364/2):]  
        
        # # weight_9 = weight_A[8, 9, :]
        # np.save('weight_A.npy', weight_A)       
        # np.save('flow_tr_2.npy', flow_tr_2)   
        
    # for j in range(N_sec-1):   # 18*125    +250 = 2500
    #     st_idx = 365-(int((j+1)*365/2)-int(j*365/2))
    #     weight_A[:, 365+int(j*365/2):365+int((j+1)*365/2)] = pred[(j+1)*N_seg:(j+2)*N_seg,st_idx:]
        
        
        
        
        if np.isnan(los):
            break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec) )
        
    # print('Epoch '+str(epoch)+': A is '+"{:.4f}".format(wghs.shape))
    
    # test on segments with training samples
    for i in range(N_sec): # better code?
        index = range(i*N_seg, (i+1)*N_seg)
        
        batch_x = x_test[index,:,:]
        batch_y = o_test[index,:]
        batch_m = mo_test[index,:]
        batch_f = flow_test[index,:]
        
        
        batch_prd = sess.run(
            [y_prd_fin],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: 1.0,
                    A_dist_up: A_dist_up_hat,
                    A_dist_dn: A_dist_down_hat,
                    A_upindex: A_upindex_hat,  
                    A_dnindex: A_dnindex_hat,
                    sflow: batch_f                    
                    
        })
        pred[index,:] = batch_prd
        
    prd_o = np.zeros([N_seg,4383])
    prd_o[:,:365] = pred[0:N_seg,:]
    
    for j in range(N_sec-1):   # 18*125    +250 = 2500
        st_idx = 365-(int((j+1)*365/2)-int(j*365/2))
        prd_o[:, 365+int(j*365/2):365+int((j+1)*365/2)] = pred[(j+1)*N_seg:(j+2)*N_seg,st_idx:]
    
    
    po = np.reshape(prd_o,[-1])
    ye = np.reshape(o_te,[-1])
    me = np.reshape(mo_te,[-1])
    rmse = np.sqrt(np.sum(np.square((po-ye)*me))/np.sum(me))
    # modify rmse to print
    print( 'Seg Test RMSE: '+"{:.4f}".format(rmse) )
    
    # np.save('predict_pde.npy',prd_o)
    np.save('multi_predict_pde_out9_5percent.npy',prd_o)
    # np.save('multi_predict_pde_5percent.npy',prd_o)
    # np.save('random_walk.npy',prd_o)
    # prd_o.shape
    # o_te.shape
    # mo_te.shape
    # np.save('o_te.npy', o_te)
    # np.save('mo_te.npy', mo_te)
    
    

    
#    # test on segments without training samples
#    me = np.reshape(mo_te_seg,[-1])
#    rmse = np.sqrt(np.sum(np.square((po-ye)*me))/np.sum(me))
#    print( 'Segwo Test RMSE: '+"{:.4f}".format(rmse) )

#     err_overall = np.multiply(prd_o - o_te, mo_te)
    
#     # err_single = np.sqrt(np.sum(np.square(err_overall[9, :])/np.sum(mo_te[9,:])))
#     # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_single) )
# #print('saving...')
# #np.save('./results/prd_RGCN_full_obstemp_cv'+str(cv_idx)+'.npy',prd_o)
#     err_2 = np.sqrt(np.sum(np.square(err_overall[2, :])/np.sum(mo_te[2,:])))
#     print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_2) )


#     err_7 = np.sqrt(np.sum(np.square(err_overall[7, :])/np.sum(mo_te[7,:])))
#     print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_7) )    
    
#     err_8 = np.sqrt(np.sum(np.square(err_overall[8, :])/np.sum(mo_te[8,:])))
#     print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_8) )    
    
#     err_9 = np.sqrt(np.sum(np.square(err_overall[9, :])/np.sum(mo_te[9,:])))
#     print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_9) )    
    
#     err_31 = np.sqrt(np.sum(np.square(err_overall[31, :])/np.sum(mo_te[31,:])))
#     print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_31) )

#     np.save('multipoints.npy',prd_o)








