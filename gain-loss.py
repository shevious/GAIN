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

  ## GAIN models
  def build_generator(self):
    x = Input(shape=(self.dim,), name='generator_input_x')
    m = Input(shape=(self.dim,), name='generator_input_m')

    a = Concatenate()([x, m])

    a = Dense(self.h_dim, activation='relu')(a)
    a = Dense(self.h_dim, activation='relu')(a)
    G_prob = Dense(self.dim, activation='sigmoid')(a)
    self.generator = keras.models.Model([x, m], G_prob, name='generator')

  def build_discriminator(self):

    x = Input(shape=(self.dim,), name='discriminator_input_x')
    h = Input(shape=(self.dim,), name='discriminator_input_h')

    a = Concatenate()([x, h])

    a = Dense(self.h_dim, activation='relu')(a)
    a = Dense(self.h_dim, activation='relu')(a)
    D_prob = Dense(self.dim, activation='sigmoid')(a)
    self.discriminator = keras.models.Model([x, h], D_prob, name='discriminator')

  def set_trainable(self, m, val):
    m.trainable = val
    for l in m.layers:
      l.trainable = val

  def G_loss_bincross(M, D_prob):
    G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
    #MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    #G_loss = G_loss_temp + self.alpha * MSE_loss 
    #return G_loss
    return G_loss_temp

  def MSE_loss(M, X, G_sample):
    MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    return MSE_loss

  def G_loss(self, M, D_prob, X, G_sample):
    #G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
    #MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    G_loss_temp = GAIN.G_loss_bincross(M, D_prob)
    MSE_loss = GAIN.MSE_loss(M, X, G_sample)
    G_loss = G_loss_temp + self.alpha * MSE_loss 
    return G_loss

  def build_adversarial(self):

    ### COMPILE DISCRIMINATOR

    self.discriminator.compile(
      optimizer=Adam(),
      #loss = 'binary_crossentropy',
      loss = self.D_loss(),
      #metrics = ['accuracy']
      metrics = ['binary_accuracy']
    )

    ### COMPILE THE FULL GAN

    self.set_trainable(self.discriminator, False)

    X = Input(shape=(self.dim,), name='model_input_x')
    M = Input(shape=(self.dim,), name='model_input_m')
    H = Input(shape=(self.dim,), name='model_input_h')

    model_input = [X, M, H]
    G_sample = self.generator([X, M])
    self.X = X

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)
    D_prob = self.discriminator([Hat_X, H])
    model_output = D_prob
    self.model = Model(model_input, model_output)

    self.model.add_loss(self.G_loss(M, D_prob, X, G_sample))
    self.model.add_metric(GAIN.G_loss_bincross(M, D_prob), name='binary_crossentropy')
    self.model.add_metric(GAIN.MSE_loss(M, X, G_sample), name='MSE_loss')

    self.model.compile( 
      optimizer=Adam(),
      #loss='binary_crossentropy',
      #loss=self.G_loss(),
      loss=None,
      metrics=['binary_accuracy']
    )
    self.model.summary()
    #print('### build')
    #print(len(self.model.inputs))
    #print(self.model.inputs[0].shape)
    #print(len(self.model.outputs))
    #print(self.model.outputs[0].shape)
    self.set_trainable(self.discriminator, True)

    '''
    self.tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
        update_freq='epoch', profile_batch=2, embeddings_freq=0,
        embeddings_metadata=None,
    )
    self.tensorboard.set_model(self.discriminator)
    '''

  # Transform train_on_batch return value
  # to dict expected by on_batch_end callback
  def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
      result[l[0]] = l[1]
    return result

  def D_loss(self):
    def loss(y_true, y_pred):
        M = y_true
        D_prob = y_pred
        ## GAIN loss
        self.D_loss = -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                      + (1-M) * tf.keras.backend.log(1. - D_prob + 1e-8)) 
        return self.D_loss
        #return { 'loss': self.D_loss, 'myloss': 0 }
    return loss

  def train_discriminator(self, M, X, H):

    ## GAIN structure

    # Generator
    #import time
    #tick = time.time()
    G_sample = self.generator.predict([X, M])
    #tock = time.time()
    #print('disc gen time = ', (tock-tick)*1000)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)

    # Discriminator
    #tick = time.time()
    return self.discriminator.train_on_batch([Hat_X, H], M)
    #tock = time.time()
    #print('disc train time = ', (tock-tick)*1000)

  def train_generator(self, X, M, H):

    ## GAIN structure

    # Generator
    #import time
    #tick = time.time()
    #self.model.train_on_batch([X, M, H], [M, X])
    #tock = time.time()
    #print('gen train time = ', (tock-tick)*1000)
    return self.model.train_on_batch([X, M, H], M)

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
  
  ## Iterations
  gain = GAIN(dim, alpha)
   
  # Start Iterations
  progress = tqdm(range(iterations))
  for it in progress:
  #for it in range(iterations):    
      
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
    d_logs = gain.train_discriminator(M_mb, X_mb, H_mb)
    #print(gain.discriminator.metrics_names)
    #print(logs)
    #gain.tensorboard.on_epoch_end(it, GAIN.named_logs(gain.discriminator, logs))
    #print('disc time = ', (tock-tick)*1000)
    D_loss_curr = gain.D_loss

    #loss = opt_G.minimize(lambda: gain.G_fun(X_mb, M_mb, H_mb), var_list = gain.theta_G)
    #tick = time.time()
    g_logs = gain.train_generator(X_mb, M_mb, H_mb)

    progress.set_description('d_loss: %f, g_loss: %f' % (d_logs[0], g_logs[0]))
    #print(gain.model.metrics_names)
    #print(logs)
    #tock = time.time()
    #print('gen time = ', (tock-tick)*1000)
    
    #G_loss_curr = gain.G_loss
    #MSE_loss_curr = gain.MSE_loss

    #_, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              #feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    #_, G_loss_curr, MSE_loss_curr = \
    #sess.run([G_solver, G_loss_temp, MSE_loss],
             #feed_dict = {X: X_mb, M: M_mb, H: H_mb})
            
  #gain.tensorboard.on_train_end(None)
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  #imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  X_mb = X_mb.astype(np.float32)
  M_mb = M_mb.astype(np.float32)
  imputed_data = gain.generator.predict([X_mb, M_mb])
  print('###')
  print(imputed_data)
  #imputed_data = imputed_data.numpy()
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data
