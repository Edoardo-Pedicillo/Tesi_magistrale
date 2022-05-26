#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cProfile import label
from queue import Empty
from turtle import shape
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1))
    # compile model
    #opt = Adadelta(learning_rate=0.1)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, layers, hamiltonian1,input):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1,input)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss
'''
def define_cost_gan(params0,params1, d_model0,d_model1, latent_dim, samples, circuit0,circuit1, nqubits, layers, hamiltonian1,epoch):
    # generate real samples
    
    #data0 = generate_training_real_samples(samples,const=0) # logistic
    #data1 = generate_training_real_samples(samples,const=1) # gamma 
    
    #x_real0, y_real0 = generate_real_samples(samples, data0, samples)
    #x_real1, y_real1 = generate_real_samples(samples, data1, samples)
    # generate fake samples
   
    x_fake0, y_fake0 = generate_fake_samples(params0, latent_dim, samples, circuit0, nqubits, layers, hamiltonian1,0)
    x_fake1, y_fake1 = generate_fake_samples(params1, latent_dim, samples, circuit1, nqubits, layers, hamiltonian1,1)
       
    # create inverted labels for the fake samples
    #y_fake0 = np.ones((samples, 1))
    #y_fake1 = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output0 = d_model0(x_fake0,training=True)
    disc_output1 = d_model1(x_fake1,training = True)
   
    loss0 = tf.keras.losses.binary_crossentropy(y_fake0, disc_output0)
    loss0 = tf.reduce_mean(loss0)
    loss1 = tf.keras.losses.binary_crossentropy(y_fake1, disc_output1)
    loss1 = tf.reduce_mean(loss1)
    
    loss_cycle0 = []
    loss_cycle1 = []
    
    for i in range(30):
        data0 = generate_training_real_samples(samples,const=0) # logistic
        data1 = generate_training_real_samples(samples,const=1) # gamma  
        # data0 are used as input for the generator0 and the output distribution is the input of generator1, data0 and the output of generator1 must be similar

        generator0_output, _ = generate_fake_samples(params0, latent_dim, samples, circuit0, nqubits, layers, hamiltonian1,0,x_input=data0)
        generator1_output, _ = generate_fake_samples(params1, latent_dim, samples, circuit1, nqubits, layers, hamiltonian1,1,x_input=generator0_output) 
        output0=generator1_output
        datamin = min(data0.min(),output0.numpy().min())
        datamax = max(data0.max(),output0.numpy().max())
        #print(datamax,datamin)
        bins=np.arange(datamin,datamax,(datamax-datamin)/20)
        bins0 = np.histogram(data0,bins=bins)
        bins_out = np.histogram(generator1_output,bins=bins)
        loss_cycle0.append(sum(abs(bins0[0]-bins_out[0]))/(2*samples))
            
        #print(2*samples)

        #print(bins0[0]-bins_out[0],sum(abs(bins0[0]-bins_out[0])))

        #print("CcCCCCCCCCCCC ",bins0)
        #print(bins_out)

        # same for data1
        generator1_output, _ = generate_fake_samples(params1, latent_dim, samples, circuit1, nqubits, layers, hamiltonian1,0,x_input=data1)
        generator0_output, _ = generate_fake_samples(params0, latent_dim, samples, circuit0, nqubits, layers, hamiltonian1,0,x_input=generator1_output) 
        output1 = generator0_output
        datamin = min(data1.min(),output1.numpy().min())
        datamax = max(data1.max(),output1.numpy().max())
        bins=np.arange(datamin,datamax,(datamax-datamin)/20)
        bins1 = np.histogram(data1,bins=bins)
        bins_out = np.histogram(generator0_output,bins=bins)
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

    loss = loss0 + loss1 + np.mean(loss_cycle0)+np.mean(loss_cycle1)
    
    loss = loss0 + loss1

    return loss
'''
def discriminator_loss(real_output, fake_output):
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss

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
    
    for q in range(nqubits):
        
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim
    circuit.set_parameters(p)  

def generate_training_real_samples(samples, const, shape=0.13):
    # generate samples from the distribution
    if const == 0: 
        x=np.random.logistic(0, shape, samples)
    elif const == 1:
        x = np.random.gamma(1.0, size=samples)
        x = x/10 - 1 
    # shape array
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
def generate_latent_points(latent_dim, samples,const):
    # generate points in the latent space
    x_input = generate_training_real_samples(latent_dim * samples,const)
    #x_input = randn(latent_dim * samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(samples, latent_dim)
    return x_input
 
# use the generator to generate fake examples, with class labels
def generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, input_distribution,x_input = []):
    # generate points in latent space
    # if input_distribution = 0 the input distribution is a logistic,
    # if input_distribution = 1 the input distribution is a gamma 

    if len(x_input) == 0 :
        
        x_input = generate_latent_points(latent_dim, samples,input_distribution)
    
    x_input = np.transpose(x_input)
    
    # generator outputs
    X = []
    # quantum generator circuit
    for i in range(samples):
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim)
        circuit_execute = circuit.execute()
        X.append(hamiltonian1.expectation(circuit_execute))
    # shape array
    X = tf.stack(X)[:, tf.newaxis]
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

def plot_data (d_loss0,d_loss1,g_loss,epoch,x_real0,x_real1,x_fake0,x_fake1):

    
    fig = plt.figure(figsize=(10,10))
    
    plt.plot(d_loss0, label= "loss discriminator 0")
    plt.plot( g_loss, label = "loss generator ")
    plt.plot(d_loss1, label= "loss discriminator 1")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig('loss_epoch_{:04d}.png'.format(epoch))
    #print(x_fake1.numpy().min())
    fig = plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    datamin = min(x_real0.min(),x_fake1.numpy().min())
    datamax = max(x_real0.max(),x_fake1.numpy().max())
       
    bins=np.arange(datamin,datamax,(datamax-datamin)/50)
    h1 = plt.hist(x_real0,bins=bins,histtype='step', color='red', label='real', alpha=0.5)
    h2 = plt.hist(x_fake1.numpy(),bins=bins,histtype='step', color = "blue" ,label='fake', alpha=0.5)

    plt.subplot(1,2,2)
    datamin = min(x_real1.min(),x_fake0.numpy().min())
    datamax = max(x_real1.max(),x_fake0.numpy().max())
       
    bins=np.arange(datamin,datamax,(datamax-datamin)/50)
    h3 = plt.hist(x_real1,bins=bins,histtype='step', color='red', label='real', alpha=0.5)
    h4 = plt.hist(x_fake0.numpy(),bins=bins,histtype='step',color='blue' , label='fake', alpha=0.5)
    plt.legend()
    plt.savefig('histogram_epoch_{:04d}.png'.format(epoch))

# train the generator and discriminator
def train( latent_dim, layers, nqubits, training_samples, discriminator0,discriminator1, circuit0,circuit1, n_epochs, samples, lr, hamiltonian1):
    d_loss0 = []
    d_loss1 = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params0 = tf.Variable(np.random.uniform(-0.15, 0.15, 4*layers*nqubits + 2*nqubits))
    initial_params1 = tf.Variable(np.random.uniform(-0.15, 0.15, 4*layers*nqubits + 2*nqubits))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    data0 = generate_training_real_samples(training_samples,const=0) # logistic
    data1 = generate_training_real_samples(training_samples,const=1) # gamma
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        print()
        x_real0, y_real0 = generate_real_samples(half_samples, data0, training_samples)
        x_real1, y_real1 = generate_real_samples(half_samples, data1, training_samples)
        
        # prepare fake examples
        x_fake0, y_fake0 = generate_fake_samples(initial_params0, latent_dim, half_samples, circuit0, nqubits, layers, hamiltonian1,0)
        x_fake1, y_fake1 = generate_fake_samples(initial_params1, latent_dim, half_samples, circuit1, nqubits, layers, hamiltonian1,1)
        #print("CCC: ",x_fake1)
        # update discriminator
        

        
        # update generator
        with tf.GradientTape() as gen_tape0,tf.GradientTape() as gen_tape1, tf.GradientTape() as disc_tape0, tf.GradientTape() as disc_tape1:
            loss0 = define_cost_gan(initial_params0,discriminator0, latent_dim, samples, circuit0 , nqubits, layers, hamiltonian1,0)
            loss1 = define_cost_gan(initial_params1,discriminator1, latent_dim, samples, circuit1, nqubits, layers, hamiltonian1,1)
            real_output0 = discriminator0(x_real1, training=True)
            fake_output0 = discriminator0(x_fake0, training=True)

            real_output1 = discriminator1(x_real0, training=True)
            fake_output1 = discriminator1(x_fake1, training=True)

            discriminator_loss0 = discriminator_loss(real_output0,fake_output0)
            discriminator_loss1 = discriminator_loss(real_output1,fake_output1)
        
        d_loss0.append( discriminator_loss0)
        d_loss1.append(discriminator_loss1)
        print(discriminator_loss0)
        
        grads = gen_tape0.gradient(loss0, initial_params0)
        optimizer.apply_gradients([(grads, initial_params0)])
        grads = gen_tape1.gradient(loss1, initial_params1)
        optimizer.apply_gradients([(grads, initial_params1)])
       
        gradients_of_discriminator0 = disc_tape0.gradient(
            discriminator_loss0, discriminator0.trainable_weights)

        gradients_of_discriminator1 = disc_tape1.gradient(
            discriminator_loss1, discriminator1.trainable_weights)

        optimizer.apply_gradients(
            zip(gradients_of_discriminator0, discriminator0.trainable_weights))
        optimizer.apply_gradients(
            zip(gradients_of_discriminator1, discriminator1.trainable_weights))
        print(initial_params0)
        '''
        gradients_of_generator0 = gen_tape0.gradient(
            loss, initial_params0)
        
        gradients_of_generator1 = gen_tape1.gradient(
            loss, initial_params1)

        gradients_of_discriminator0 = disc_tape0.gradient(
            discriminator_loss0, discriminator0.trainable_variables)

        gradients_of_discriminator1 = disc_tape1.gradient(
            discriminator_loss1, discriminator1.trainable_variables)

        optimizer.apply_gradients(
            zip([gradients_of_generator0], [initial_params0]))
        optimizer.apply_gradients(
            zip([gradients_of_generator1], [initial_params1]))
        optimizer.apply_gradients(
            zip(gradients_of_discriminator0, discriminator0.trainable_variables))
        optimizer.apply_gradients(
            zip(gradients_of_discriminator1, discriminator1.trainable_variables))
        '''
        g_loss.append(loss0+loss1)

        if i%2 == 0 :
            x_real0, y_real0 = generate_real_samples(half_samples, data0, training_samples)
            x_real1, y_real1 = generate_real_samples(half_samples, data1, training_samples)
            
            # prepare fake examples
            x_fake0, y_fake0 = generate_fake_samples(initial_params0, latent_dim, half_samples, circuit0, nqubits, layers, hamiltonian1,0)
            x_fake1, y_fake1 = generate_fake_samples(initial_params1, latent_dim, half_samples, circuit1, nqubits, layers, hamiltonian1,1)

            plot_data (d_loss0,d_loss1,g_loss,i,x_real0,x_real1,x_fake0,x_fake1)

        #np.savetxt(f"PARAMS_1Dgamma_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_epochs}", [initial_params.numpy()], newline='')
        #np.savetxt(f"dloss_1Dgamma_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_epochs}", [d_loss], newline='')
        #np.savetxt(f"gloss_1Dgamma_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_epochs}", [g_loss], newline='')
        # serialize weights to HDF5
        #discriminator.save_weights(f"discriminator_1Dgamma_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_epochs}.h5")

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
    # create quantum generator 1
    circuit1 = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit1.add(gates.RY(q, 0))
            circuit1.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit1.add(gates.RY(q, 0)) 
    # create quantum generator 2
    circuit2 = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit2.add(gates.RY(q, 0))
            circuit2.add(gates.RZ(q, 0))
    for q in range(nqubits):
        circuit2.add(gates.RY(q, 0))   
    # create classical discriminator 1
    #print(circuit1.get_parameters())
    discriminator1 = define_discriminator()
    # create classical discriminator 2
    discriminator2 = define_discriminator()
    # train model
    train(latent_dim, layers, nqubits, training_samples, discriminator1,discriminator2, circuit1,circuit2, n_epochs, batch_samples, lr, hamiltonian1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=0, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--n_epochs", default=2000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.5, type=float)
    args = vars(parser.parse_args())
    main(**args)
