# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:50 2019

@author: xiaoweijia
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
import math
# run only using CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

''' Declare constants '''
learning_rate = 0.001 #0.005
learning_rate_pre = 0.005
epochs = 50
epochs_pre = 0 #200#100#200#70
batch_size = 2000
state_size = 20 
input_size = 20-10-3
static_size = 3
phy_size = 2
T = 13149
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
npic = 12
n_steps = int(4383/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1
N_seg = 42
kb=1.0

n_upd = 5
n_fupd = 5
upd_lr = 0.005

s_perc = 1.0

''' Build Graph '''
tf.reset_default_graph()
random.seed(9001)
# Graph input/output
x = tf.placeholder("float", [None, n_steps, None]) #tf.float32
x_static = tf.placeholder("float", [None, n_steps, None]) #tf.float32
# sflow = tf.placeholder("float", [None, n_steps])

y = tf.placeholder("float", [None, n_steps]) #tf.int32
m = tf.placeholder("float", [None, n_steps])

A = tf.placeholder("float", [N_seg,N_seg])  

W_e1 = tf.get_variable('W_e1',[input_size, state_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b_e1 = tf.get_variable('b_e1',[state_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

W_e2 = tf.get_variable('W_e2',[static_size, state_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b_e2 = tf.get_variable('b_e2',[state_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

W_e3 = tf.get_variable('W_e3',[state_size, state_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b_e3 = tf.get_variable('b_e3',[state_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

W_c1 = tf.get_variable('W_c1',[state_size, state_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b_c1 = tf.get_variable('b_c1',[state_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

W_c2 = tf.get_variable('W_c2',[static_size, state_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b_c2 = tf.get_variable('b_c2',[state_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))
                

w_p = tf.get_variable('w_p',[state_size, n_classes], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b_p = tf.get_variable('b_p',[n_classes],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))


def forward(x, x_static, A,
            W_e1,b_e1,W_e2,b_e2,W_c1,b_c1, W_c2, b_c2, 
            w_p, b_p, reuse=False):
    
    # output = []
    E = []
    
#    C_gr = tf.nn.tanh(tf.matmul(A,tf.matmul(E_pre,W_c1)+b_c1)+ (tf.matmul(x_static,W_c2)+b_c2) )
    E_pre = tf.nn.tanh( (tf.matmul(x[:,0,:],W_e1)+b_e1)+(tf.matmul(x_static[:,0,:], W_e2)+b_e2) )
    E.append(E_pre)
            
            
    for t in range(1,n_steps):
        
        C_gr = tf.nn.tanh(tf.matmul(A,tf.matmul(E_pre,W_c1)+b_c1)+ (tf.matmul(x_static[:,t,:] ,W_c2)+b_c2) )
        E_pre = tf.nn.tanh( (tf.matmul(x[:,t,:],W_e1)+b_e1)+(tf.matmul(x_static[:,t,:],W_e2)+b_e2)+(tf.matmul(C_gr,W_e3)+b_e3) )
        
        E.append(E_pre)
        
    E = tf.stack(E,axis=1)
    Ep = tf.reshape(E,[-1,state_size])
    
    
    pred = tf.matmul(Ep,w_p)+b_p
    pred = tf.reshape(pred,[-1,n_steps,1])

    return Ep, pred

def loss_measure(pred,y,m):
    pred_s = tf.reshape(pred,[-1,1])
    y_s = tf.reshape(y,[-1,1])
    m_s = tf.reshape(m,[-1,1])
    r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s-y_s),m_s)))/tf.reduce_sum(m_s))
    return r_cost



h_tr,pred_tr = forward(x, x_static, A,
            W_e1,b_e1,W_e2,b_e2,W_c1,b_c1, W_c2, b_c2, 
            w_p, b_p,)

cost = loss_measure(pred_tr,y,m)


saver = tf.train.Saver(max_to_keep=3)

tvars = tf.trainable_variables()
for i in tvars:
    print(i)

#tvars = [Wg2, bg2, Ui,Uf,Uo,Ug,Us,Wi,Wf,Wo,Wg,Ws]
gr = tf.gradients(cost, tvars)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(gr, tvars))





''' Load data '''
feat = np.load('processed_features.npy')
obs = np.load('obs_temp.npy') #np.load('obs_temp.npy')
maso = (obs!=-11).astype(int)


#feat = np.delete(feat,[9,10],2)
feat = np.delete(feat,[1,2,6,7,8,9,10,12,14,15],2)

# dates = np.load('dates.npy') 
adj_up = np.load('up_full.npy') 
adj_dn = np.load('dn_full.npy')


adj = adj_up #+adj_dn
mean_adj = np.mean(adj[adj!=0])
std_adj = np.std(adj[adj!=0])
adj[adj!=0] = adj[adj!=0]-mean_adj
adj[adj!=0] = adj[adj!=0]/std_adj
adj[adj!=0] = 1/(1+np.exp(adj[adj!=0]))

#A_hat = adj.copy() + I
A_hat = adj.copy()
A_hat[A_hat==np.nan]=0
D = np.sum(A_hat, axis=1)
D[D==0]=1
D_inv = D**-1.0
D_inv = np.diag(D_inv)
A_hat = np.matmul(D_inv,A_hat)

I = np.eye(adj.shape[0])
A_hat = A_hat + I


x_te = feat[:,cv_idx*4383:(cv_idx+1)*4383,:]
o_te = obs[:,cv_idx*4383:(cv_idx+1)*4383]
mo_te = maso[:,cv_idx*4383:(cv_idx+1)*4383]





if cv_idx==1:
    x_tr_1 = feat[:,:4383,:]
    o_tr_1 = obs[:,:4383]
    mo_tr_1 = maso[:,:4383]
    
    x_tr_2 = feat[:,2*4383:3*4383,:]
    o_tr_2 = obs[:,2*4383:3*4383:]
    mo_tr_2 = maso[:,2*4383:3*4383:] 

if cv_idx==2:
    x_tr_1 = feat[:,:4383,:]
    o_tr_1 = obs[:,:4383]
    mo_tr_1 = maso[:,:4383]
    
    x_tr_2 = feat[:,4383:2*4383,:]
    o_tr_2 = obs[:,4383:2*4383:]
    mo_tr_2 = maso[:,4383:2*4383:]



### sparsify mask

# mo_tr_1 = np.reshape(mo_tr_1,[-1])
# mo_tr_2 = np.reshape(mo_tr_2,[-1])
# indices = np.random.choice(np.arange(4383*N_seg), replace=False, size=int(4383*N_seg * (1-s_perc)))
# mo_tr_1[indices] = 0.0
# indices = np.random.choice(np.arange(4383*N_seg), replace=False, size=int(4383*N_seg * (1-s_perc)))
# mo_tr_2[indices] = 0.0

# mo_tr_1 = np.reshape(mo_tr_1,[N_seg,4383])
# mo_tr_2 = np.reshape(mo_tr_2,[N_seg,4383])

# ts = np.sum(mo_tr_1)+np.sum(mo_tr_2)
# print(ts)







# s_perc = 0.05  #0.2#0.4 #0.6 #0.8
# #ts = np.sum(m_tr_1[seg_test,:])+np.sum(m_tr_2[seg_test,:])

# indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
# mo_tr_1[:,indices] = 0.0
# indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
# mo_tr_2[:,indices] = 0.0


### data on cold seasons
for i in range(0, 365*12):        
    if math.floor((i - 243)/365) > math.floor((i - 334)/365) :
        mo_tr_1[:, i] = 0.0

for i in range(0, 365*12):        
    if math.floor((i - 243)/365) > math.floor((i - 334)/365) :
        mo_tr_2[:, i] = 0.0

for i in range(0, 365*12):        
    if math.floor((i - 243)/365) == math.floor((i - 334)/365) :
        mo_te[:, i] = 0.0   



x_train_1 = np.zeros([N_seg*N_sec,n_steps,10]) #use 10 instead of input size, input size == 7
o_train_1 = np.zeros([N_seg*N_sec,n_steps])
mo_train_1 = np.zeros([N_seg*N_sec,n_steps])

x_train_2 = np.zeros([N_seg*N_sec,n_steps,10])
o_train_2 = np.zeros([N_seg*N_sec,n_steps])
mo_train_2 = np.zeros([N_seg*N_sec,n_steps])

x_test = np.zeros([N_seg*N_sec,n_steps,10])
o_test = np.zeros([N_seg*N_sec,n_steps])
mo_test = np.zeros([N_seg*N_sec,n_steps])

for i in range(1,N_sec+1):
    x_train_1[(i-1)*N_seg:i*N_seg,:,:]=x_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    o_train_1[(i-1)*N_seg:i*N_seg,:]=o_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_1[(i-1)*N_seg:i*N_seg,:]=mo_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    x_train_2[(i-1)*N_seg:i*N_seg,:,:]=x_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    o_train_2[(i-1)*N_seg:i*N_seg,:]=o_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_2[(i-1)*N_seg:i*N_seg,:]=mo_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    x_test[(i-1)*N_seg:i*N_seg,:,:]=x_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    o_test[(i-1)*N_seg:i*N_seg,:]=o_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_test[(i-1)*N_seg:i*N_seg,:]=mo_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]




 
        
        
        
''' Session starts '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())


print('Fine-tuning starts')
print('==================================')
#total_batch = int(np.floor(N_tr/batch_size))
los = 0
mre = 10
pred = np.zeros(o_test.shape)
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
        
        batch_x = x_train_1[index,:,0:7]
        batch_x_static = x_train_1[index,:,7:]
        batch_y = o_train_1[index,:]
        batch_m = mo_train_1[index,:]

#        
        if np.sum(batch_m)>0:
            _,los_s = sess.run(
                [train_op, cost],
                feed_dict = {
                        x: batch_x,
                        x_static: batch_x_static,
                        y: batch_y,
                        m: batch_m,
                        A: A_hat
            })
            alos += los
            alos_s += los_s
            if np.isnan(los):
                break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec))
          # +': loss_p '+"{:.4f}".format(alos_p/N_sec) )
    
    if np.isnan(los):
        break
        
    alos = 0
    alos_s = 0
    alos_p = 0
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_2[index,:,0:7]
        batch_x_static = x_train_2[index,:,7:]
        batch_y = o_train_2[index,:]
        batch_m = mo_train_2[index,:]

        if np.sum(batch_m)>0:
            _,los_s = sess.run(
                [train_op, cost],
                feed_dict = {
                        x: batch_x,
                        x_static: batch_x_static,
                        y: batch_y,
                        m: batch_m,
                        A: A_hat
            })
            alos += los
            alos_s += los_s
            if np.isnan(los):
                break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec))
          # +': loss_p '+"{:.4f}".format(alos_p/N_sec) )
    
    
# test on segments with training samples
prd_te = np.zeros([N_sec*N_seg,n_steps,1])

for i in range(N_sec): # better code?
    index = range(i*N_seg, (i+1)*N_seg)
    
    batch_x = x_test[index,:,0:7]
    batch_x_static = x_test[index,:,7:]
    batch_y = o_test[index,:]
    batch_m = mo_test[index,:]
    
    batch_prd = sess.run(
        pred_tr,
        feed_dict = {
                x: batch_x,
                x_static: batch_x_static,
                y: batch_y,
                m: batch_m,
                A: A_hat
    })
    prd_te[index,:,:] = batch_prd
    
prd_o = np.zeros([N_seg,4383])
prd_o[:,:365] = prd_te[0:N_seg,:,0]

for j in range(N_sec-1):   # 18*125    +250 = 2500
    st_idx = 365-(int((j+1)*365/2)-int(j*365/2))
    prd_o[:, 365+int(j*365/2):365+int((j+1)*365/2)] = prd_te[(j+1)*N_seg:(j+2)*N_seg,st_idx:,0]


po = np.reshape(prd_o,[-1])
ye = np.reshape(o_te,[-1])
me = np.reshape(mo_te,[-1])
rmse = np.sqrt(np.sum(np.square((po-ye)*me))/np.sum(me))
print( 'Seg Test RMSE: '+"{:.4f}".format(rmse) )



    # err_overall = np.multiply(prd_o - o_te, mo_te)
    
    # # err_single = np.sqrt(np.sum(np.square(err_overall[31, :])/np.sum(m_te[31,:])))
    # # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_single) )
    
    # err_2 = np.sqrt(np.sum(np.square(err_overall[2, :])/np.sum(mo_te[2,:])))
    # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_2) )


    # err_7 = np.sqrt(np.sum(np.square(err_overall[7, :])/np.sum(mo_te[7,:])))
    # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_7) )    
    
    # err_8 = np.sqrt(np.sum(np.square(err_overall[8, :])/np.sum(mo_te[8,:])))
    # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_8) )    
    
    # err_9 = np.sqrt(np.sum(np.square(err_overall[9, :])/np.sum(mo_te[9,:])))
    # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_9) )    
    
    # err_31 = np.sqrt(np.sum(np.square(err_overall[31, :])/np.sum(mo_te[31,:])))
    # print( 'Single Seg Test RMSE: '+"{:.4f}".format(err_31) )        



    
    
    
    
    
    









