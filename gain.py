'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index

from tensorflow import keras

class GAIN():
  def __init__(self, dim, alpha):
    self.dim = dim
    self.alpha = alpha
    self.h_dim = int(dim)

    # Discriminator variables
    self.D_W1 = tf.Variable(xavier_init([self.dim*2, self.h_dim])) # Data + Hint as inputs
    self.D_b1 = tf.Variable(tf.zeros(shape = [self.h_dim]))
    
    self.D_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
    self.D_b2 = tf.Variable(tf.zeros(shape = [self.h_dim]))
    
    self.D_W3 = tf.Variable(xavier_init([self.h_dim, self.dim]))
    self.D_b3 = tf.Variable(tf.zeros(shape = [self.dim]))  # Multi-variate outputs
    
    self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    self.G_W1 = tf.Variable(xavier_init([self.dim*2, self.h_dim]))  
    self.G_b1 = tf.Variable(tf.zeros(shape = [self.h_dim]))
    
    self.G_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
    self.G_b2 = tf.Variable(tf.zeros(shape = [self.h_dim]))
    
    self.G_W3 = tf.Variable(xavier_init([self.h_dim, self.dim]))
    self.G_b3 = tf.Variable(tf.zeros(shape = [self.dim]))
    
    self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

  ## GAIN functions
  # Generator
  def generator(self, x, m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(self, x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
    D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

  @tf.function
  def D_fun(self, M, X, H):
    self.M = M
    self.X = X
    self.H = H

    ## GAIN structure
    # Generator
    self.G_sample = self.generator(X, M)

    # Combine with observed data
    self.Hat_X = X * M + self.G_sample * (1-M)
    
    # Discriminator
    self.D_prob = self.discriminator(self.Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.keras.backend.log(self.D_prob + 1e-8) \
                                  + (1-M) * tf.keras.backend.log(1. - self.D_prob + 1e-8)) 

    self.D_loss = D_loss_temp
    return self.D_loss
    
  @tf.function
  def G_fun(self, X, M, H):
    ## GAIN structure
    # Generator
    G_sample = self.generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)

    # Discriminator
    D_prob = self.discriminator(Hat_X, H)

    G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
    
    self.MSE_loss = \
    tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    self.G_loss = G_loss_temp + self.alpha * self.MSE_loss 
    return self.G_loss
    

def gain (data_x, gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  print('###')
  print('data_x = ', data_x)
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  norm_data_x = norm_data_x.astype(np.float32)
  data_x = data_x.astype(np.float32)
  
  opt_D = tf.keras.optimizers.Adam()
  opt_G = tf.keras.optimizers.Adam()

  gain = GAIN(dim, alpha)
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

    X_mb = X_mb.astype(np.float32)
    M_mb = M_mb.astype(np.float32)
    H_mb = H_mb.astype(np.float32)
    #print('###')
    #print('X_mb.ndtype = ', X_mb.dtype)
    #print('H_mb.ndtype = ', H_mb.dtype)
    #print('M_mb.ndtype = ', M_mb.dtype)
    #print('Z_mb.ndtype = ', Z_mb.dtype)
    #print('norm_data_x.ndtype = ', norm_data_x.dtype)

    import time
    tick = time.time()
    loss = opt_D.minimize(lambda: gain.D_fun(M_mb, X_mb, H_mb), var_list = gain.theta_D)
    D_loss_curr = gain.D_loss
    tock = time.time()
    print('D = ', (tock-tick)*1000)

    tick = time.time()
    loss = opt_G.minimize(lambda: gain.G_fun(X_mb, M_mb, H_mb), var_list = gain.theta_G)
    tock = time.time()
    print('G = ', (tock-tick)*1000)
    
    G_loss_curr = gain.G_loss
    MSE_loss_curr = gain.MSE_loss

    #_, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              #feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    #_, G_loss_curr, MSE_loss_curr = \
    #sess.run([G_solver, G_loss_temp, MSE_loss],
             #feed_dict = {X: X_mb, M: M_mb, H: H_mb})
            
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  #imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  X_mb = X_mb.astype(np.float32)
  M_mb = M_mb.astype(np.float32)
  imputed_data = gain.generator(X_mb, M_mb)
  imputed_data = imputed_data.numpy()
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data
