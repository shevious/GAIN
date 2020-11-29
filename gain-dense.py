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

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
import tensorflow
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

#tf.config.optimizer.set_jit(True)

class GDense(tf.keras.layers.Layer):
  def __init__(self, w, b):
    super(GDense, self).__init__()
    self.w = w
    self.b = b

  def call(self, x):
    return tf.matmul(x, self.w) + self.b

class GAIN():
  def __init__(self, dim, alpha):
    self.dim = dim
    self.alpha = alpha
    self.h_dim = int(dim)

    # Discriminator variables
    self.D_W1 = tf.Variable(xavier_init([self.dim*2, self.h_dim]), trainable=True) # Data + Hint as inputs
    self.D_b1 = tf.Variable(tf.zeros(shape = [self.h_dim]), trainable=True)
    
    self.D_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
    self.D_b2 = tf.Variable(tf.zeros(shape = [self.h_dim]))
    
    self.D_W3 = tf.Variable(xavier_init([self.h_dim, self.dim]))
    self.D_b3 = tf.Variable(tf.zeros(shape = [self.dim]))  # Multi-variate outputs
    
    self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    self.G_W1 = tf.Variable(xavier_init([self.dim*2, self.h_dim]), trainable=True)  
    self.G_b1 = tf.Variable(tf.zeros(shape = [self.h_dim]), trainable=True)
    
    self.G_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]), trainable=True)
    self.G_b2 = tf.Variable(tf.zeros(shape = [self.h_dim]), trainable=True)
    
    self.G_W3 = tf.Variable(xavier_init([self.h_dim, self.dim]))
    self.G_b3 = tf.Variable(tf.zeros(shape = [self.dim]))
    
    self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]
    self.build_generator()
    self.generator.summary()
    self.build_discriminator()
    self.discriminator.summary()
    self.build_adversarial()
    #self.model.summary()

  def build_generator_org(self):
    x = Input(shape=(self.dim,), name='generator_input_x')
    m = Input(shape=(self.dim,), name='generator_input_m')
    inputs = Concatenate()([x, m])
    #print(x)
    #print(m)
    #print(inputs)
    #print('G_W1 = ', self.G_W1)
    #g_w1 = K.expand_dims(self.G_W1, 0) # shape (114,57) ==> (1,114,57)
    #g_w2 = K.expand_dims(self.G_W2, 0) # shape (57,57) ==> (1,57,57)
    #g_w3 = K.expand_dims(self.G_W3, 0) # shape (57,57) ==> (1,57,57)
    #g_b1 = K.expand_dims(self.G_b1, 0) # shape (114,57) ==> (1,114,57)
    #g_b2 = K.expand_dims(self.G_b2, 0) # shape (57,57) ==> (1,57,57)
    #g_b3 = K.expand_dims(self.G_b3, 0) # shape (57,57) ==> (1,57,57)
    #a = Dot(axes=(1,0))([inputs, self.G_W1])
    a = GDense(w = self.G_W1, b = self.G_b1)(inputs)
    #G_h1 = ReLU()(Add()([Dot(axes=(1,0))([inputs, K.expand_dims(self.G_W1],0)), self.G_b1]))
    a = ReLU()(a) # G_h1

    #a = Dot(axes=1)([a, g_w2])
    #a = Add()([a, g_b2])
    a = GDense(w = self.G_W2, b = self.G_b2)(a)
    a = ReLU()(a) # G_h2
    a = GDense(w = self.G_W3, b = self.G_b3)(a)
    G_prob = Activation('sigmoid')(a) 
    self.generator = keras.models.Model([x, m], G_prob, name='generator')

  def build_generator(self):
    x = Input(shape=(self.dim,), name='generator_input_x')
    m = Input(shape=(self.dim,), name='generator_input_m')

    inputs = Concatenate()([x, m])

    a = Dense(self.h_dim, activation='relu')(inputs)
    a = Dense(self.h_dim, activation='relu')(a)
    G_prob = Dense(self.dim, activation='sigmoid')(a)
    self.generator = keras.models.Model([x, m], G_prob, name='generator')

  def build_discriminator(self):

    x = Input(shape=(self.dim,), name='discriminator_input_x')
    h = Input(shape=(self.dim,), name='discriminator_input_h')

    inputs = Concatenate()([x, h])

    a = Dense(self.h_dim, activation='relu')(inputs)
    a = Dense(self.h_dim, activation='relu')(a)
    D_prob = Dense(self.dim, activation='sigmoid')(a)
    self.discriminator = keras.models.Model([x, h], D_prob, name='discriminator')

  ## GAIN functions
  # Generator
  def generator_fun(self, x, m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator_fun(self, x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
    D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

  def set_trainable(self, m, val):
    m.trainable = val
    for l in m.layers:
      l.trainable = val

  def build_adversarial(self):

    ### COMPILE DISCRIMINATOR

    self.discriminator.compile(
      optimizer=Adam(),
      loss = 'binary_crossentropy',
      #loss = self.D_loss(),
      metrics = ['accuracy']
    )

    self.generator.compile(
      optimizer=Adam(),
      loss = 'binary_crossentropy',
      #loss = self.D_loss(),
      metrics = ['accuracy']
    )

    ### COMPILE THE FULL GAN

    self.set_trainable(self.discriminator, False)
    #model_input = Input(shape=(self.dim,), name='model_input')
    X = Input(shape=(self.dim,), name='model_input_x')
    M = Input(shape=(self.dim,), name='model_input_m')
    H = Input(shape=(self.dim,), name='model_input_h')
    model_input = [X, M, H]
    G_sample = self.generator([X, M])

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)
    D_prob = self.discriminator([Hat_X, H])
    model_output = [D_prob, G_sample]
    #model_output = self.discriminator(self.generator(model_input))
    self.model = Model(model_input, model_output)

    self.model.compile( 
      optimizer=Adam(),
      #loss='binary_crossentropy',
      loss=self.G_loss(),
      metrics=['accuracy']
    )
    self.model.summary()
    #print('### build')
    #print(len(self.model.inputs))
    #print(self.model.inputs[0].shape)
    #print(len(self.model.outputs))
    #print(self.model.outputs[0].shape)
    self.set_trainable(self.discriminator, True)

  def D_loss(self):
    @tf.function
    def loss(y_true, y_pred):
        M = y_true
        D_prob = y_pred
        ## GAIN loss
        self.D_loss = -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                      + (1-M) * tf.keras.backend.log(1. - D_prob + 1e-8)) 
        return self.D_loss
    return loss

  def G_loss(self):
    @tf.function
    def loss(y_true, y_pred):
        #print('y_pred.shape =', y_pred.shape)
        #D_prob, G_sample = y_pred
        D_prob = y_pred
        #M, X = y_true
        M = y_true
        self.G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
        #MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
        #self.G_loss = G_loss_temp + self.alpha * self.MSE_loss 
        return self.G_loss_temp
    return loss

  def train_discriminator(self, M, X, H):

    ## GAIN structure

    # Generator
    import time
    tick = time.time()
    G_sample = self.generator.predict([X, M])
    tock = time.time()
    print('disc gen time = ', (tock-tick)*1000)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)

    # Discriminator
    tick = time.time()
    self.discriminator.train_on_batch([Hat_X, H], M)
    tock = time.time()
    print('disc train time = ', (tock-tick)*1000)

  def train_generator(self, X, M, H):

    ## GAIN structure

    # Generator
    return self.model.train_on_batch([X, M, H], [M, X])

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

    G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(self.D_prob + 1e-8))
    
    self.MSE_loss = \
    tf.reduce_mean((M * X - M * self.G_sample)**2) / tf.reduce_mean(M)

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
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  norm_data_x = norm_data_x.astype(np.float32)
  data_x = data_x.astype(np.float32)
  
  '''
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = keras.Input(shape = [dim])
  # Mask vector 
  M = keras.Input(shape = [dim])
  # Hint vector
  H = keras.Input(shape = [dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                                + (1-M) * tf.keras.backend.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  #D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  #G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  #adam_D_solver = tf.keras.optimizers.Adam()
  #D_solver = adam_D_solver.minimize(D_loss, var_list=theta_D)
  #G_solver = tf.keras.optimizers.Adam().minimize(G_loss, var_list=theta_G)
  '''
  
  ## Iterations
  #sess = tf.Session()
  #sess.run(tf.global_variables_initializer())
  opt_D = tf.keras.optimizers.Adam()
  opt_G = tf.keras.optimizers.Adam()
  #D_fun = lambda M, X, H: D_loss
  #D_fun = tf.keras.backend.function(inputs=[M,X,H], outputs=D_loss)
  gain = GAIN(dim, alpha)
   
  # Start Iterations
  #for it in tqdm(range(iterations)):    
  for it in range(iterations):    
      
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

    #loss = opt_D.minimize(lambda: gain.D_fun(M_mb, X_mb, H_mb), var_list = gain.theta_D)
    import time
    gain.train_discriminator(M_mb, X_mb, H_mb)
    #print('disc time = ', (tock-tick)*1000)
    D_loss_curr = gain.D_loss

    #loss = opt_G.minimize(lambda: gain.G_fun(X_mb, M_mb, H_mb), var_list = gain.theta_G)
    tick = time.time()
    gain.train_generator(X_mb, M_mb, H_mb)
    tock = time.time()
    print('gen time = ', (tock-tick)*1000)
    
    G_loss_curr = gain.G_loss
    #MSE_loss_curr = gain.MSE_loss

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
