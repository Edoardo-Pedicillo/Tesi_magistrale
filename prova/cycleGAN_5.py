#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
# train a quantum-classical generative adversarial network on LHC data
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten
from qibo import gates, hamiltonians, models, set_backend, set_threads
import matplotlib.pyplot as plt
import argparse

set_backend('tensorflow')
set_threads(4)

# define the standalone discriminator model
def define_discriminator(n_inputs=1, alpha=0.2, dropout=0.2):
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
def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, layers, hamiltonian1):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

def define_cycle_cost_gan(params0,params1, latent_dim, samples, circuit0,circuit1, nqubits, layers, hamiltonian1 ):
    """
        here is defined the condition of cycle gan
    """
    loss_cycle0 = []
    loss_cycle1 = []
    for i in range(1):
        data1 = np.random.logistic(0,0.13,latent_dim*(latent_dim*samples))#generate_training_real_samples(samples*latent_dim,input=0) # logistic
        #data1 = generate_training_real_samples(samples*latent_dim,input=1) # gamma 
        data0 = np.random.gamma(1., size=(latent_dim*samples)*latent_dim)
        data0 = data0/4 - 1
        data0=np.reshape(data0,(latent_dim*samples,latent_dim) )
        data1=np.reshape(data1,(latent_dim*samples,latent_dim) )

        #print("data0",data0.shape)
        # data0 are used as input for the generator0 and the output distribution is the input of generator1, data0 and the output of generator1 must be similar
        #print(samples)
        generator0_output, _ = generate_fake_samples(params0, latent_dim, samples*latent_dim, circuit0, nqubits, layers, hamiltonian1,0,x_input=data0)
        #print(type(generator0_output))
        if i == 0:
            plt.figure(figsize=(10,10))
            
            h1=plt.hist(generator0_output.numpy(), histtype='step', color='red', label='input', alpha=0.5)
            
                
            plt.legend() 
            plt.savefig("generator0_output.png")

        generator0_output = generator0_output.numpy().reshape(samples,latent_dim)
        #tf.reshape(generator0_output,[samples,latent_dim]) 
        generator1_output, _ = generate_fake_samples(params1, latent_dim, samples, circuit1, nqubits, layers, hamiltonian1,1,x_input=generator0_output) 
        output0=generator1_output
        
        data0=np.reshape(data0,(latent_dim**2,samples) )[0]
        #print(len(data0))
        datamin = min(data0.min(),output0.numpy().min())
        datamax = max(data0.max(),output0.numpy().max())
        #print(datamax,datamin)
        bins=np.arange(datamin,datamax,(datamax-datamin)/20)
        bins0 = np.histogram(data0,bins=bins)
        bins_out = np.histogram(output0.numpy(),bins=bins)
        loss_cycle0.append(sum(abs(bins0[0]-bins_out[0]))/(2*samples))
        # same for data1
        generator1_output, _ = generate_fake_samples(params1, latent_dim, samples*latent_dim, circuit1, nqubits, layers, hamiltonian1,0,x_input=data1)
        generator1_output = generator1_output.numpy().reshape(samples,latent_dim)
        generator0_output, _ = generate_fake_samples(params0, latent_dim, samples, circuit0, nqubits, layers, hamiltonian1,0,x_input=generator1_output) 
        output1 = generator0_output

        data1=np.reshape(data1,(latent_dim**2,samples))[0]

        datamin = min(data1.min(),output1.numpy().min())
        datamax = max(data1.max(),output1.numpy().max())
        bins=np.arange(datamin,datamax,(datamax-datamin)/20)
        bins1 = np.histogram(data1,bins=bins)
        bins_out = np.histogram(output1,bins=bins)
        loss_cycle1.append(sum(abs(bins1[0]-bins_out[0]))/(2*samples))

        if i == 0:
            plt.figure(figsize=(10,10))
            plt.subplot(1,2,1)
            h1=plt.hist(data0, bins=bins0[1],histtype='step', color='red', label='input', alpha=0.5)
            h2=plt.hist(output0.numpy(),bins=bins0[1],histtype='step', color='blue', label='cycle output', alpha=0.5)
                
            plt.legend() 
            plt.subplot(1,2,2)
            h3=plt.hist(data1, bins=bins1[1], histtype='step', color='red', label='input', alpha=0.5)
            h4=plt.hist(output1.numpy(),bins=bins1[1], histtype='step', color='blue', label='cycle output', alpha=0.5)

            plt.legend()
            plt.savefig('histogram_cycle.png')

    return np.mean(loss_cycle0)+np.mean(loss_cycle1)

def set_params(circuit, params, x_input, i, nqubits, layers, latent_dim):
    p = []
    index = 0
    noise = 0
    #print(params,index)
    for l in range(layers):
        for q in range(nqubits):
           # print("H",i)
            p.append(params[index]*x_input[noise][i] + params[index+1])
            
            index+=2
            noise=(noise+1)%latent_dim
            #print(noise)
            #print("G ",index,x_input[noise][i])
            p.append(params[index]*x_input[noise][i] + params[index+1])
            #print("L")
            index+=2
            noise=(noise+1)%latent_dim
            #print("C ",index)
    for q in range(nqubits):
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim
        #print("D ", index)
    #print(len(p))
    circuit.set_parameters(p)  

def generate_training_real_samples(samples, shape=0.13,input=0):
    # generate samples from the distribution
    #x = np.random.gamma(shape, size=samples)
    #x = x/4 - 1
    if input == 0:
        x=np.random.logistic(0, shape, samples)
    elif input == 1:
        x = np.random.gamma(1., size=samples)
        x = x/4 - 1
   
    x = np.reshape(x, (samples,1))
    return x
 
 
# generate real samples with class labels
def generate_real_samples(samples, distribution, real_samples):
    # generate samples from the distribution
    idx = np.random.randint(real_samples, size=samples)
    X = distribution[idx,:]
    # generate class labels
    y = np.ones((samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, samples,input):
    # generate points in the latent space
    if input == 0:
        x_input = np.random.gamma(1.0, size=latent_dim*samples)
        x_input = x_input/4 - 1
    elif input == 1:
        x_input=np.random.logistic(0, 0.13, latent_dim*samples)
    #x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
   #print("AAAAAA ",x_input.shape)
    return x_input
 
# use the generator to generate fake examples, with class labels
def generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1,input=0,x_input=[]):
    # generate points in latent space

    
    if len(x_input) == 0:
        #print("prova2")
        x_input = generate_latent_points(latent_dim, samples,input)
    #x_input = x_input.reshape(samples, latent_dim)
    x_input = np.transpose(x_input)
    # generator outputs
    X = []
    # quantum generator circuit
    #print("x_input",x_input.shape)
    #print(samples)
    for i in range(samples):
        #print("pppppppppp: ",x_input)
        #print("sampl ", samples,x_input.shape)
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim)
        circuit_execute = circuit.execute()
        X.append(hamiltonian1.expectation(circuit_execute))
    # shape array
    X = tf.stack(X)[:, tf.newaxis]
    # create class labels
    y = np.zeros((samples, 1))
    return X, y


# train the generator and discriminator
def train(d_model, d_model1, latent_dim, layers, nqubits, training_samples, circuit,circuit1, n_epochs, samples, lr, hamiltonian1):
    d_loss = []
    g_loss = []
    d_loss1 = []
   
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(-0.15, 0.15, 4*layers*nqubits + 2*nqubits))
    initial_params1 = tf.Variable(np.random.uniform(-0.15, 0.15, 4*layers*nqubits + 2*nqubits))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s = generate_training_real_samples(training_samples,input = 0)

    t = generate_training_real_samples(training_samples,input = 1) 
    start = time.time()
    tp=[]
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        print()
        
        x_real, y_real = generate_real_samples(half_samples, s, training_samples)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, circuit, nqubits, layers, hamiltonian1,input=0)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        
        # serialize weights to HDF5
        #discriminator.save_weights(f"discriminator_1Dgamma_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")

        
        x_real1, y_real1 = generate_real_samples(half_samples, t, training_samples)
        # prepare fake examples
        x_fake1, y_fake1 = generate_fake_samples(initial_params1, latent_dim, half_samples, circuit1, nqubits, layers, hamiltonian1,input=1)
        # update discriminator
        d_loss_real1, _ = d_model1.train_on_batch(x_real1, y_real1)
        d_loss_fake1, _ = d_model1.train_on_batch(x_fake1, y_fake1)
        d_loss1.append((d_loss_real1 + d_loss_fake1)/2)
        # update generator
        with tf.GradientTape() as tape1, tf.GradientTape() as tape:
            
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples, circuit, nqubits, layers, hamiltonian1) +define_cost_gan(initial_params1, d_model1, latent_dim, samples, circuit1, nqubits, layers, hamiltonian1)+10*define_cycle_cost_gan(initial_params,initial_params1, latent_dim, samples, circuit,circuit1, nqubits, layers, hamiltonian1 )
            
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        grads1 = tape1.gradient(loss, initial_params1)
        optimizer.apply_gradients([(grads1, initial_params1)])
        g_loss.append(loss)

        tp.append(time.time()-start)
        np.savetxt(f"PARAMS_cGAN_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [initial_params.numpy()], newline='')
        np.savetxt(f"dloss_cGAN_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [d_loss], newline='')
        np.savetxt(f"gloss_cGAN_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [g_loss], newline='')
        np.savetxt(f"PARAMS1_cGAN_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [initial_params.numpy()], newline='')
        np.savetxt(f"dloss1_cGAN_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [d_loss1], newline='')
        np.savetxt(f"time_cGAN_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [tp], newline='')
        #np.savetxt(f"gloss_cGAN1_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}", [g_loss], newline='')
        # serialize weights to HDF5
        #discriminator.save_weights(f"discriminator_1Dgamma_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")

def main(latent_dim, layers, training_samples, n_epochs, batch_samples, lr):
    
    # define hamiltonian to generate fake samples
    def hamiltonian1():
        m0 = hamiltonians.Z(1).matrix
        ham = hamiltonians.Hamiltonian(1, m0)
        return ham
    
    # number of qubits generator
    nqubits = 1
    # create hamiltonians
    hamiltonian1 = hamiltonian1()
    # create quantum generator
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))   
    circuit1 = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit1.add(gates.RY(q, 0))
            circuit1.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit1.add(gates.RY(q, 0))   
    # create classical discriminator
    discriminator = define_discriminator()
    discriminator1 = define_discriminator()
    # train model
    train(discriminator, discriminator1,latent_dim, layers, nqubits, training_samples, circuit, circuit1, n_epochs, batch_samples, lr, hamiltonian1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--n_epochs", default=30000, type=int)
    parser.add_argument("--batch_samples", default=328, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    args = vars(parser.parse_args())
    main(**args)
