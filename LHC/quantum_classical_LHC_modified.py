#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from main import readInit, readEvent, GetEnergySquared, GetMandelT, GetRapidity
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import argparse

set_backend('tensorflow')
set_threads(4)

tot_params=34

def load_fixed_params(n_params=0):

    fparams=np.loadtxt("fixed_PARAMS", delimiter=' ')

    return fparams[:int(n_params)]

# define the standalone discriminator model
def define_discriminator(n_inputs=3, alpha=0.2, dropout=0.2):
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
def define_cost_gan(params, discriminator, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3, n_params):
    # generate fake samples
    x_fake, y_fake = generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3,n_params)
    # create inverted labels for the fake samples
    y_fake = np.ones((samples, 1))
    # evaluate discriminator on fake examples
    disc_output = discriminator(x_fake)
    loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
    loss = tf.reduce_mean(loss)
    return loss

def set_params(circuit, params, x_input, i, nqubits, layers, latent_dim,n_params):
    p = []
    index = 0
    noise = 0
    
    fparams=load_fixed_params(n_params)


    for _ in range(int(len(fparams)/2)):
        p.append(fparams[index]*x_input[noise][i] + fparams[index+1])
        index+=2
        noise=(noise+1)%latent_dim
    
    index=0
    len_params=tot_params-len(fparams)
    
    for _ in range(int(len_params/2)):
       
        #print(index,params[index],params[index+1])
        p.append(params[index]*x_input[noise][i] + params[index+1])
        index+=2
        noise=(noise+1)%latent_dim

    circuit.set_parameters(p) 

def load_events(filename,samples=20000):
    init = readInit(filename)
    evs = list(readEvent(filename))

    invar = np.zeros((len(evs),3))
    for ev in range(len(evs)):
         invar[ev, 0] = GetEnergySquared(evs[ev])
         invar[ev, 1] = GetMandelT(evs[ev])
         invar[ev, 2] = GetRapidity(init, evs[ev])
         
    return invar[:samples, :]
 
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
def generate_fake_samples(params, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3, n_params):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, samples)
    x_input = np.transpose(x_input)
    # generator outputs
    X1 = []
    X2 = []
    X3 = []
    # quantum generator circuit
    for i in range(samples):
        set_params(circuit, params, x_input, i, nqubits, layers, latent_dim,n_params)
        circuit_execute = circuit.execute()
        X1.append(hamiltonian1.expectation(circuit_execute))
        X2.append(hamiltonian2.expectation(circuit_execute))
        X3.append(hamiltonian3.expectation(circuit_execute))
    # shape array
    X = tf.stack((X1, X2, X3), axis=1)
    # create class labels
    y = np.zeros((samples, 1))
    return X, y

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

   
    return sum(prob_real[i] * np.log(prob_real[i]/prob_fake[i]) for i in range(len(prob_real)))

# train the generator and discriminator
def train(d_model, latent_dim, layers, nqubits, training_samples, discriminator, circuit, n_epochs, samples, lr, hamiltonian1, hamiltonian2, hamiltonian3,n_params,iterator):
    bins=100
    d_loss = []
    g_loss = []
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    initial_params = tf.Variable(np.random.uniform(-0.15, 0.15, tot_params-n_params))
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s = load_events('data/ppttbar_10k_events.lhe', training_samples)
    init = readInit('data/ppttbar_10k_events.lhe')
    evs = list(readEvent('data/ppttbar_10k_events.lhe'))    
    invar = np.zeros((len(evs),3))
    # manually enumerate epochs

    for ev in range(len(evs)):
        invar[ev, 0] = GetEnergySquared(evs[ev])
        invar[ev, 1] = GetMandelT(evs[ev])
        invar[ev, 2] = GetRapidity(init, evs[ev])         
            
    pt = PowerTransformer()
    print(pt.fit(invar[:10000, :]))
    print(pt.lambdas_)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(scaler.fit(pt.transform(invar[:10000, :])))
    x_real = load_events('data/ppttbar_10k_events.lhe')
    x_real1 = []
    x_real2 = []
    x_real3 = []

    for j in range(len(x_real)):
        x_real1.append(x_real[j][0])
        x_real2.append(x_real[j][1])
        x_real3.append(x_real[j][2])

    
    for i in range(n_epochs):
        
        # prepare real samples
        x_real, y_real = generate_real_samples(half_samples, s, training_samples)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(initial_params, latent_dim, half_samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3,n_params)
        # update discriminator
        d_loss_real, _ = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, _ = d_model.train_on_batch(x_fake, y_fake)
        d_loss.append((d_loss_real + d_loss_fake)/2)
        # update generator
        with tf.GradientTape() as tape:
            loss = define_cost_gan(initial_params, d_model, latent_dim, samples, circuit, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3,n_params)
        grads = tape.gradient(loss, initial_params)
        optimizer.apply_gradients([(grads, initial_params)])
        g_loss.append(loss)

        if i%25==0: # Changeee
            
            print(i)
            x_fake,_ = generate_fake_samples(initial_params, latent_dim, 3000, circuit, nqubits, layers, hamiltonian1, hamiltonian2, hamiltonian3,n_params)
            
           
            x_fake = pt.inverse_transform(scaler.inverse_transform(x_fake))
            x_fake1 = []
            x_fake2 = []
            x_fake3 = []
           

            for j in range(len(x_fake)):
                x_fake1.append(x_fake[j][0])
                x_fake2.append(x_fake[j][1])
                x_fake3.append(x_fake[j][2])
            
            
            bins_real=np.histogram(x_real1, bins = bins)
            bins_fake=np.histogram(x_fake1, bins = bins_real[1])
            #print(x_fake1)
            #print(x_real1)
            #print(bins_real,bins_fake)
            kl1=kl_divergence(bins_real[0],bins_fake[0],epsilon=0.1)
            
            bins_real=np.histogram(x_real2, bins = bins)
            bins_fake=np.histogram(x_fake2, bins = bins_real[1])
            kl2=kl_divergence(bins_real[0],bins_fake[0],epsilon=0.1)
            
            bins_real=np.histogram(x_real3, bins = bins)
            bins_fake=np.histogram(x_fake3, bins = bins_real[1])
            kl3=kl_divergence(bins_real[0],bins_fake[0],epsilon=0.1)
            
            if i != 0:
                
                with open(f"KLdiv_LHC_s_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", "ab") as f:
                    
                    np.savetxt(f, [kl1], newline=' ')
                
                with open(f"KLdiv_LHC_t_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", "ab") as f:
                    
                    np.savetxt(f, [kl2], newline=' ')
                
                with open(f"KLdiv_LHC_y_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", "ab") as f:
                    
                    np.savetxt(f, [kl3], newline=' ')
                
                with open(f"bins_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", "ab") as f:
                    
                    np.savetxt(f, bins_fake[0], newline=' ')
            else:
                np.savetxt(f"KLdiv_LHC_s_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", [kl1], newline=' ')
                np.savetxt(f"KLdiv_LHC_t_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", [kl2], newline=' ')
                np.savetxt(f"KLdiv_LHC_y_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", [kl3], newline=' ')
                np.savetxt(f"bins_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}_{iterator}", bins_fake[0], newline=' ')

        # np.savetxt(f"PARAMS_transfer_learning_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}", [initial_params.numpy()], newline='')
        # np.savetxt(f"dloss_transfer_learning_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}", [d_loss], newline='')
        # np.savetxt(f"gloss_transfer_learning_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}", [g_loss], newline='')
        # serialize weights to HDF5
        #discriminator.save_weights(f"discriminator_LHCdata_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{n_params}.h5")

def main(latent_dim, layers, training_samples, n_epochs, batch_samples, lr,n_params,iterator):
    
    # define hamiltonian to generate fake samples
    def hamiltonian1():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(id, np.kron(id, m0))
        ham = hamiltonians.Hamiltonian(3, m0)
        return ham
    
    def hamiltonian2():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(id, np.kron(m0, id))
        ham = hamiltonians.Hamiltonian(3, m0)
        return ham
    
    def hamiltonian3():
        id = [[1, 0], [0, 1]]
        m0 = hamiltonians.Z(1).matrix
        m0 = np.kron(m0, np.kron(id, id))
        ham = hamiltonians.Hamiltonian(3, m0)
        return ham
    
    # number of qubits generator
    nqubits = 3
    # create hamiltonians
    hamiltonian1 = hamiltonian1()
    hamiltonian2 = hamiltonian2()
    hamiltonian3 = hamiltonian3()
    # create quantum generator
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, 0))
            circuit.add(gates.RZ(q, 0))
        if l==1 or l==5 or l==9 or l==13 or l==17:
            circuit.add(gates.CRY(0, 1, 0))
            circuit.add(gates.CRY(0, 2, 0))
        if l==3 or l==7 or l==11 or l==15 or l==19:
            circuit.add(gates.CRY(1, 2, 0))
            circuit.add(gates.CRY(2, 0, 0))
    for q in range(nqubits):
        circuit.add(gates.RY(q, 0))   
    # create classical discriminator
    discriminator = define_discriminator()
    # train model
    train(discriminator, latent_dim, layers, nqubits, training_samples, discriminator, circuit, n_epochs, batch_samples, lr, hamiltonian1, hamiltonian2, hamiltonian3,n_params,iterator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--training_samples", default=10000, type=int)
    parser.add_argument("--n_epochs", default=30000, type=int)
    parser.add_argument("--batch_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--n_params", default=0, type=int)
    parser.add_argument("--iterator", default=0, type=int)
    args = vars(parser.parse_args())
    main(**args)
