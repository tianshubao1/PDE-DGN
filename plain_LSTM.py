# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:42:07 2020

@author: tians
"""

from __future__ import print_function, division
import numpy as np
# import tensorflow as tf

# from tensorflow.python.framework import ops
# ops.reset_default_graph()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


import random
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

''' Declare constants '''
learning_rate = 0.01
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 200    #70
batch_size = 2000       #ignore this
hidden_size = 20 
input_size = 20-2
phy_size = 2        #physical variable
T = 13149               #13149 time points
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
npic = 12       #time series 
n_steps = int(4383/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1        #sliding window, so more sequences
N_seg = 42                  # number of rivers
kb=1.0                      # dropout, no dropout


''' build model '''
# model = Sequential()
# model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# model.add(LSTM(50, activation='relu'))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

tf.reset_default_graph()


# tf.compat.v1.reset_default_graph()
random.seed(9001)
# Graph input/output
x = tf.placeholder("float", [None, n_steps,input_size]) #tf.float32
p = tf.placeholder("float", [None, n_steps,phy_size])
y = tf.placeholder("float", [None, n_steps]) #tf.int32
m = tf.placeholder("float", [None, n_steps])
#A = tf.placeholder("float", [N_seg,N_seg])

# A_dist = tf.placeholder("float", [42, 2])
# A_upindex = tf.placeholder("float", [42, 42])
# A_dnindex = tf.placeholder("float", [42, 42])
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




# W2 = tf.get_variable('W_2',[hidden_size, n_classes], tf.float32,
#                                  tf.random_normal_initializer(stddev=0.02))
# b2 = tf.get_variable('b_2',[n_classes],  tf.float32,
#                                  initializer=tf.constant_initializer(0.0))

# lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0) 
# state_series_x, current_state_x = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32) #46*(500*7)
# h = tf.reshape(state_series_x,[-1,hidden_size])
# h = tf.nn.dropout(h,keep_prob)
# y_prd = tf.matmul(h,W2)+b2







# lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0) 

lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0) 
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0) 
o_sr = []
o_phy = []

output_series, current_state = lstm_cell(x[:,0,:], lstm_cell.zero_state(N_seg, dtype=tf.float32)) #46*(500*7)


c_pre,h_pre = current_state
# state_graph = tf.contrib.rnn.LSTMStateTuple(c_pre,h_pre)
state_graph = tf.nn.rnn_cell.LSTMStateTuple(c_pre,h_pre)
o_sr.append(h_pre)
o_phy.append(tf.matmul(h_pre,Wp)+bp)


identity = tf.eye(42)        
for t in range(1,n_steps):  #river other segments

        output_series, current_state = lstm_cell(x[:,t,:], state_graph) #46*(500*7)
      
        c,h = current_state
        # h_gr = tf.nn.tanh(tf.matmul(h_pre,Wg1)+bg1)
        # # whether transform the state as well
        # c_gr = tf.nn.tanh(tf.matmul(c_pre,Wg2)+bg2)
        
        
        # h_pre = tf.nn.sigmoid(tf.matmul(h,Wa1)+tf.matmul(h_gr,Wa2)+ba)
        # c_pre = tf.nn.sigmoid(tf.matmul(c,Wc1)+tf.matmul(c_gr,Wc2)+bc)
                
        h_pre = tf.nn.sigmoid(tf.matmul(h,Wa1)+ba)
        c_pre = tf.nn.sigmoid(tf.matmul(c,Wc1)+bc)
        
        # state_graph = tf.contrib.rnn.LSTMStateTuple(c_pre,h_pre)
        state_graph = tf.nn.rnn_cell.LSTMStateTuple(c_pre,h_pre)
        o_sr.append(h_pre)
        o_phy.append(tf.matmul(h_pre,Wp)+bp)     #all out put phy
    
    
    
o_sr = tf.stack(o_sr,axis=1) # N_seg - T - hidden_size
oh = tf.reshape(o_sr,[-1,hidden_size])
o_phy = tf.stack(o_phy,axis=1)
o_phy = o_phy[:,:,:phy_size]
o_phy = tf.reshape(o_phy,[-1])


cost_phy = tf.sqrt(tf.reduce_sum(tf.square(o_phy-tf.reshape(p,[-1])))/(phy_size*N_seg*n_steps))  #p is true value
#intermidia


oh = tf.nn.dropout(oh,keep_prob)

y_prd = tf.matmul(oh,W2)+b2

#  cost function
y_prd_fin = tf.reshape(y_prd,[-1,n_steps])
y_prd = tf.reshape(y_prd,[-1])
cost_sup = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((y_prd-tf.reshape(y,[-1])),tf.reshape(m,[-1]))))/(tf.reduce_sum(tf.reshape(m,[-1]))+1))


tvars = tf.trainable_variables()
for i in tvars:
    print(i)
saver = tf.train.Saver(max_to_keep=3)

#this part is pre train
optimizer_pre = tf.train.AdamOptimizer(learning_rate_pre)
gvs_pre = optimizer_pre.compute_gradients(cost_phy)
train_op_pre = optimizer_pre.apply_gradients(gvs_pre)


#this part is train
optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(cost_sup)
train_op = optimizer.apply_gradients(gvs)








''' load data '''
feat = np.load('processed_features.npy')
label = np.load('sim_temp.npy') #np.load('obs_temp.npy')
obs = np.load('obs_temp.npy') #np.load('obs_temp.npy')
mask = (label!=-11).astype(int)
maso = (obs!=-11).astype(int)

seg_test = np.load('sel_test_id.npy')

#seg_test = np.load('sel_seg_hard.npy')
flow = np.load('sim_flow.npy')
phy = np.concatenate([np.expand_dims(label,2),np.expand_dims(flow,2)],axis=2)  #concatenate label and flow
phy = np.reshape(phy,[-1,phy_size])


from sklearn import preprocessing
phy = preprocessing.scale(phy)
phy = np.reshape(phy,[N_seg,-1,phy_size])

feat = np.delete(feat,[9,10],2)


x_te = feat[:,cv_idx*4383:(cv_idx+1)*4383,:] #12 years, 365 days, cv = cross validation, test
y_te = label[:,cv_idx*4383:(cv_idx+1)*4383]
o_te = obs[:,cv_idx*4383:(cv_idx+1)*4383]
m_te = mask[:,cv_idx*4383:(cv_idx+1)*4383]      #simulation mask
mo_te = maso[:,cv_idx*4383:(cv_idx+1)*4383]         #observation mask 
flow_te = flow[:,cv_idx*4383:(cv_idx+1)*4383] 



if cv_idx==1:
    x_tr_1 = feat[:,:4383,:]    # one 2d plane
    y_tr_1 = label[:,:4383]
    o_tr_1 = obs[:,:4383]
    m_tr_1 = mask[:,:4383]
    mo_tr_1 = maso[:,:4383]
    mo_tr_1[31, :] = 0
    flow_tr_1 = flow[:,:4383]
    
    x_tr_2 = feat[:,2*4383:3*4383,:]
    y_tr_2 = label[:,2*4383:3*4383:]
    o_tr_2 = obs[:,2*4383:3*4383:]
    m_tr_2 = mask[:,2*4383:3*4383:]
    mo_tr_2 = maso[:,2*4383:3*4383:] 
    mo_tr_2[31, :] = 0
    flow_tr_2 = flow[:,2*4383:3*4383:]
    
    
if cv_idx==2:
    x_tr_1 = feat[:,:4383,:]
    y_tr_1 = label[:,:4383]
    o_tr_1 = obs[:,:4383]
    m_tr_1 = mask[:,:4383]
    mo_tr_1 = maso[:,:4383]
    mo_tr_1[31, :] = 0
    flow_tr_1 = flow[:,:4383]
    
    
    x_tr_2 = feat[:,4383:2*4383,:]
    y_tr_2 = label[:,4383:2*4383:]
    o_tr_2 = obs[:,4383:2*4383:]
    m_tr_2 = mask[:,4383:2*4383:]
    mo_tr_2 = maso[:,4383:2*4383:]
    mo_tr_2[31, :] = 0
    flow_tr_2 = flow[:,4383:2*4383:]





x_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size])
y_train_1 = np.zeros([N_seg*N_sec,n_steps])
o_train_1 = np.zeros([N_seg*N_sec,n_steps])
m_train_1 = np.zeros([N_seg*N_sec,n_steps])
mo_train_1 = np.zeros([N_seg*N_sec,n_steps])
flow_train_1 = np.zeros([N_seg*N_sec,n_steps])

x_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size])
y_train_2 = np.zeros([N_seg*N_sec,n_steps])
o_train_2 = np.zeros([N_seg*N_sec,n_steps])
m_train_2 = np.zeros([N_seg*N_sec,n_steps])
mo_train_2 = np.zeros([N_seg*N_sec,n_steps])
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
    y_train_1[(i-1)*N_seg:i*N_seg,:]=y_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_train_1[(i-1)*N_seg:i*N_seg,:]=o_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_1[(i-1)*N_seg:i*N_seg,:]=m_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_1[(i-1)*N_seg:i*N_seg,:]=mo_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    flow_train_1[(i-1)*N_seg:i*N_seg,:]=flow_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    
    x_train_2[(i-1)*N_seg:i*N_seg,:,:]=x_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_train_2[(i-1)*N_seg:i*N_seg,:]=y_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_train_2[(i-1)*N_seg:i*N_seg,:]=o_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_2[(i-1)*N_seg:i*N_seg,:]=m_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_2[(i-1)*N_seg:i*N_seg,:]=mo_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    flow_train_2[(i-1)*N_seg:i*N_seg,:]=flow_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    
    x_test[(i-1)*N_seg:i*N_seg,:,:]=x_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_test[(i-1)*N_seg:i*N_seg,:]=y_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_test[(i-1)*N_seg:i*N_seg,:]=o_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_test[(i-1)*N_seg:i*N_seg,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_test[(i-1)*N_seg:i*N_seg,:]=mo_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
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
        batch_f = flow_train_1[index,:]

      
        _,los_s = sess.run(
            [train_op, cost_sup],          #input different
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    sflow: batch_f                     
                    
        })
        alos += los
        alos_s += los_s
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
        batch_f = flow_train_2[index,:]
        _,los_s = sess.run(
            [train_op, cost_sup],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    sflow: batch_f                     
        })
        alos += los
        alos_s += los_s
        if np.isnan(los):
            break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec) )
    
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
    
    # np.save('rnn_pred.npy',prd_o)    
    np.save('rnn_pred_out31.npy',prd_o)    











