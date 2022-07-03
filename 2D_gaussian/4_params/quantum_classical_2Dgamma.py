#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train a quantum-classical generative adversarial network on LHC data
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten
from qibo import gates, hamiltonians, models, set_backend, set_threads
import argparse

set_backend('tensorflow')

def kl_divergence(bins_real, bins_fake,epsilon):
    
    epsilon=0.1
    prob_real=[]
    prob_fake=[]
    for i in range (len(bins_real)):
        prob_real.append(bins_real[i]+epsilon)
        prob_fake.append(epsilon+bins_fake[i])

    #print(prob_fake,prob_real)  

    prob_real=prob_real/sum(prob_real) # probability for each bin (Normalization)
    prob_fake=prob_fake/sum(prob_fake)

   
    return sum(prob_real[i] * np.log(prob_real[i]/prob_fake[i]) for i in range(len(prob_real)))# Convergence problem if a[i] or b[i] equals zero. 
                                                            #I add a little quantity to each bin to avoid problems
   
# define the standalone discriminator model
def define_discriminator(n_inputs=2, alpha=0.2, dropout=0.2):
    model = Sequential()
        
    model.add(Dense(200, use_bias=False, input_dim=n_inputs))
    model.add(Reshape((10,10,2)))
    
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2D(8, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal'))

    model.add(Flatten())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout)) 

    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    opt = Adadelta(learning_rate=0.1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

def set_params(circuit, params, x_input, i, nqubits, layers, latent_dim):
    p = []
    index = 0
    noise = 0
    
    for l in range(layers):
        for q in range(nqubits):
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
            p.append(params[index]*x_input[noise][i] + params[index+1])
            index+=2
            noise=(noise+1)%latent_dim
            #print( " 1 " , index)
        #p.append(params[index]*x_input[noise][i] + params[index+1])
        #index+=2
        #noise=(noise+1)%latent_dim
        
        # print(" 2" ,index)
    for q in range(nqubits):
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim
        #print(q," 3 " ,index)
    circuit.set_parameters(p) 

def generate_training_real_samples(samples):
  # generate training samples from the distribution
    s = []
  
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]        
    x, y = np.random.multivariate_normal(mean, cov, samples).T/4
    s1 = np.reshape(x, (samples,1))
    s2 = np.reshape(y, (samples,1))
    '''
    s1 = np.random.gamma(1., size=samples)
    s1 = s1/2 - 1
    s1= np.reshape(s1, (samples,1))
    
    s2 = np.random.gamma(1., size=samples)
    s2 = s2/2 - 1
    s2= np.reshape(s2, (samples,1))
    '''
    s = np.hstack((s1,s2))
    return s
 
# generate real samples with class labels
def generate_real_samples(samples, distribution, real_samples):
    # generate samples from the distribution
    idx = np.random.randint(real_samples, size=samples)
    X = distribution[idx,:]
    # generate class labels
    y = np.ones((samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, samples):
    # generate points in the latent space
    x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
    return x_input
 
# use the generator to generate fake examples, with class labels
def generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    x_input = np.transpose(x_input)
    # generator outputs
    X1 = []
    X2 = []
    # quantum generator circuit
    for i in range(samples):
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim)
        circuit_execute = circuit.execute()
        X1.append(hamiltonian1.expectation(circuit_execute))
        X2.append(hamiltonian2.expectation(circuit_execute))
        
    # shape array
    X = tf.stack((X1, X2), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

# train the generator and discriminator
def train(d_model, latent_dim, layers, nqubits, training_samples, discriminator, circuit, n_epochs, samples, lr, hamiltonian1, hamiltonian2,iterator):
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(-0.15, 0.15, 4))#4*layers*nqubits + 2*nqubits + 2*layers))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s = generate_training_real_samples(training_samples)
    start=time.time()
    t = []
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_samples, s, training_samples)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)
       
        if i%25==0:
            data=5000
            training_data=5000
            x_real_kl,_ = generate_real_samples(data, s, training_data)
        
            x_fake_kl,_ = generate_fake_samples(initial_params, latent_dim, data, circuit, nqubits, layers, hamiltonian1, hamiltonian2)
            print(f"number of epochs {i}")
            hh_real = np.histogram2d(tf.transpose(x_real_kl)[0], tf.transpose(x_real_kl)[1] , bins=40)
            hh_fake = np.histogram2d(tf.transpose(x_fake_kl)[0], tf.transpose(x_fake_kl)[1] , bins=hh_real[1])
            
            if i != 0:

                with open(f"KLdiv_2Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{iterator}", "ab") as f:
                    
                    np.savetxt(f, [kl_divergence(hh_real[0].flatten(),hh_fake[0].flatten() ,epsilon=0.01)], newline=' ')
            
            else:
                np.savetxt(f"KLdiv_2Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{iterator}", [kl_divergence(hh_real[0].flatten(),hh_fake[0].flatten() ,epsilon=0.01)], newline=' ')
            
        np.savetxt(f"PARAMS_2Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{iterator}", [initial_params.numpy()], newline='')
        np.savetxt(f"dloss_2Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{iterator}", [d_loss], newline='')
        np.savetxt(f"gloss_2Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{iterator}", [g_loss], newline='')
        # serialize weights to HDF5
       # discriminator.save_weights(f"discriminator_2Dgaussian_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")
        t.append(time.time() - start )
        np.savetxt(f"time_2Dgaussian_gamma_different_circuit_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{iterator}", [t], newline='')
def main(latent_dim, layers, training_samples, n_epochs, batch_samples, lr,iterator):
    
    # define hamiltonian to generate fake samples
    def hamiltonian1():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(id, m0)
        ham = hamiltonians.Hamiltonian(2, m0)
        return ham
    
    def hamiltonian2():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(m0, id)
        ham = hamiltonians.Hamiltonian(2, m0)
        return ham
    
    
    # number of qubits generator
    nqubits = 2
    # create hamiltonians
    hamiltonian1 = hamiltonian1()
    hamiltonian2 = hamiltonian2()
   
    # create quantum generator
    circuit = models.Circuit(nqubits)
    
            
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))   
    # create classical discriminator
    discriminator = define_discriminator()
    # train model
    train(discriminator, latent_dim, layers, nqubits, training_samples, discriminator, circuit, n_epochs, batch_samples, lr, hamiltonian1, hamiltonian2,iterator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--n_epochs", default=30000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.5, type=float)
    parser.add_argument("--iterator", default=0, type=int) 
    args = vars(parser.parse_args())
    main(**args)
