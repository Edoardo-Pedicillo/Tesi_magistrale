import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Reshape, LeakyReLU, Flatten
from qibo import gates, models, set_backend
import argparse

set_backend('tensorflow')

class generator():
    
    def __init__(self, latent_dim,
                    layers, nqubits,
                    lr, avg, dataset, 
                    digit, pixels,batch_samples,params=[] ) -> None:

        # digit is the digit input of the generator
        self.latent_dim = latent_dim
        self.nqubits = nqubits
        self.lr = lr
        self.avg = avg
        self.dataset = dataset
        self.digit = digit
        self.layers = layers 
        self.pixels = pixels
        self.batch_samples = batch_samples
        self.circuit = self.set_circuit()
        self.conv_generator = self.define_conv_generator(n_inputs=pixels, lr=lr)
        if len(params) == 0:
            self.params = tf.Variable(np.random.uniform(-0.15, 0.15, 4*layers*nqubits + 2*nqubits))
        else:
            self.params = tf.Variable(params)
            print("parameters uploaded")
        self.loss = []

    def set_circuit (self):
        
        circuit = models.Circuit(self.nqubits)
        for l in range(self.layers):
            for i in range(self.nqubits):
                circuit.add(gates.RY(i, 0))
            for i in range(0, self.nqubits - 1, 2):
                circuit.add(gates.CZ(i, i + 1))
            for i in range(self.nqubits):
                circuit.add(gates.RY(i, 0))
            for i in range(1, self.nqubits - 2, 2):
                circuit.add(gates.CZ(i, i + 1))
            circuit.add(gates.CZ(0, self.nqubits - 1))
        for q in range(self.nqubits):
            circuit.add(gates.RY(q, 0))
        
        return circuit

    def set_params(self, x_input, i):
        p = []
        index = 0
        noise = 0
        for l in range(self.layers):
            for q in range(self.nqubits):
                p.append(self.params[index] * x_input[noise][i] + self.params[index+1])
                index += 2
                noise= (noise+1) % self.latent_dim
                p.append(self.params[index] * x_input[noise][i] + self.params[index+1])
                index += 2
                noise = (noise+1) % self.latent_dim
        for q in range(self.nqubits):
            p.append(self.params[index] * x_input[noise][i] + self.params[index+1])
            index += 2
            noise= (noise+1) % self.latent_dim
        self.circuit.set_parameters(p)
    
    def define_conv_generator(self, n_inputs=16, alpha=0.2, dropout=0.2, lr=0.1):
        
        model = Sequential()
        model.add(Conv2D(128, (2, 2), strides=(2, 2), padding='same',
                                        ))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(
            32, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
    

        model.add(Dense(3*64, activation="relu"))
        model.build([int(self.batch_samples/2),8,8,1])
        return model


    def generate_fake_samples(self,samples = 0, x_input = [] ):

        if len(x_input) == 0:
        
            x_input,_=generate_training_samples(self.dataset,  samples, self.avg, self.digit) 

        else: 

            samples = len(x_input)

        X = []
        for i in range(self.pixels):
            X.append([])
        
        # quantum generator circuit
        images_input=tf.reshape(x_input,[samples,8,8,1])
       
        circuit_inputs=self.conv_generator(images_input,training=True)
        
        for k,j in enumerate(circuit_inputs):
            
            j=tf.reshape(j,[3,64])
            j=tf.cast(j,tf.double)
            self.set_params( j, k)
            self.circuit()
            for ii in range(self.pixels):
                    X[ii].append(abs(self.circuit.final_state[ii]**2))

        # shape array
        X = tf.stack([X[i] for i in range(len(X))], axis=1)
        # create class labels
        y = np.zeros((samples, 1))
        return X, y

    def define_cost_gan( self, discriminator,samples):
        # generate fake samples
        x_fake, _ = self.generate_fake_samples(samples)
        # create inverted labels for the fake samples
        y_fake = np.ones((samples, 1))
        # evaluate discriminator on fake examples
        disc_output = discriminator.dis(x_fake)
        loss = tf.keras.losses.binary_crossentropy(y_fake, disc_output)
        loss = tf.reduce_mean(loss)
        return loss

class discriminator():

    def __init__(self, lr,  pixels ) -> None:
        
            self.lr = lr 
            self.pixels = pixels
            self.loss = []
            self.dis = self.define_discriminator()

    def define_discriminator(self, alpha=0.2, dropout=0.2):
        n_inputs = self.pixels
        lr = self.lr
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
        opt = Adadelta(learning_rate=lr)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def generate_real_samples(self,X_train,samples):
        # generate samples from the distribution
        idx = np.random.randint(X_train.shape[0], size=samples)
        imgs = X_train[idx]
        y = np.ones((samples, 1))

        return imgs, y


def generate_training_samples(dataset, samples, avg=1, digit=0):
    if dataset == 'dummy_dataset.npy':
        pixels = 16
        images = np.load('dummy_dataset.npy')
        images = images[:samples]

    elif dataset == '4x4MNIST':
        from scipy.io import loadmat
        pixels = 16
        mat = loadmat('4x4MNIST_Train/4x4MNIST_Train/MNIST_Train_Nox4x4.mat')
        images = mat['V'][:samples]
    elif dataset == '8x8Digits':
        from sklearn.datasets import load_digits
        pixels = 64
        images, y = load_digits(n_class=2, return_X_y=True)
        images = images[np.where(y==digit)][:samples]
        images = images.astype(np.float32) / 16.
    elif dataset == 'MNIST':
        from keras.datasets import mnist
        (images, _), (_, _) = mnist.load_data()
        pixels = 784
        images = images[:samples]
        images = images.astype(np.float32) / 255.
    elif dataset == 'Lund':
        images = np.load("reference_images.npy")
        pixels = 512
        images = images[:samples]
        # rotation
        images = np.array( [ i [::,::-1].T for i in images])
        # average
        averager = Averager(avg)
        images = averager.transform(images)
        condition = generate_condition()
        reduced = []
        for i in range(len(images)):
            masked =  np.ma.masked_array(images[i], mask=False)
            masked.mask = condition
            reduced.append(np.ma.compressed(masked))
    else:
        return NotImplementedError("Unknown dataset")

    if dataset == 'Lund':
        #normalize each image:
        reduced = np.array([i/sum(i.flatten()) for i in reduced])
        # # # reshape
        images = np.reshape(reduced, (reduced.shape[0], pixels))
    else:
        #normalize each image:
        images = np.array([i/sum(i.flatten()) for i in images])
        # # # reshape
        images = np.reshape(images, (images.shape[0], pixels))
    return images, pixels

def dloss(d_model,gen,s,samples):
    x_real, y_real = d_model.generate_real_samples(s, samples)
    # prepare fake examples
    x_fake, y_fake = gen.generate_fake_samples(samples)
   
    # update discriminator
    d_loss_real, _ = d_model.dis.train_on_batch(x_real, y_real)
    d_loss_fake, _ = d_model.dis.train_on_batch(x_fake, y_fake)
    d_model.loss.append((d_loss_real + d_loss_fake)/2)

def cycle_loss( gen0, gen1, samples, digit, dataset, avg  ):

    input_images, _ = generate_training_samples(dataset, samples, avg, digit)
    
    output_gen0, _ = gen0.generate_fake_samples(x_input=input_images)

    output_images, _ = gen1.generate_fake_samples(x_input = output_gen0)
   
    return tf.reduce_mean(tf.abs(input_images - output_images)).numpy()

# train the generator and discriminator
def train(d_model, d_model1, gen, gen1, latent_dim, layers, nqubits, training_samples, n_epochs, samples, lr, lr_d, avg, dataset,
          folder):
    
    # determine half the size of one batch, for updating the discriminator
    half_samples = int(samples / 2)
    optimizer = tf.optimizers.Adadelta(learning_rate=lr)
    # prepare real samples
    s, _ = generate_training_samples(dataset, training_samples, avg, 0)

    t, _ = generate_training_samples(dataset, training_samples, avg, 1)
    
    # manually enumerate epochs
    for i in range(n_epochs):

        print("epoch ",i)
        
        # discriminator loss 
        dloss(d_model,gen,s,half_samples)
        dloss(d_model1,gen1,t,half_samples)
        
       
        with tf.GradientTape() as tape,\
            tf.GradientTape() as tape1,\
            tf.GradientTape() as tape2,\
            tf.GradientTape() as tape3:

            loss = (gen.define_cost_gan( d_model,samples)
                + gen1.define_cost_gan (d_model1, samples) 
                + cycle_loss( gen, gen1, samples, 1, dataset, avg  )
                + cycle_loss( gen1, gen, samples, 0, dataset, avg  )
            )
        
        # update generator gen     
        grads = tape.gradient(loss,gen.params)
        optimizer.apply_gradients([(grads, gen.params)])
       
        gradients_conv_generator = tape1.gradient(
            loss, gen.conv_generator.trainable_variables)
       
        optimizer.apply_gradients(
            zip(gradients_conv_generator, gen.conv_generator.trainable_variables))

        gen.loss.append(loss)

        # update generator gen1  
        grads = tape2.gradient(loss,gen1.params)
        optimizer.apply_gradients([(grads, gen1.params)])
       
        gradients_conv_generator = tape3.gradient(
            loss, gen1.conv_generator.trainable_variables)
       
        optimizer.apply_gradients(
            zip(gradients_conv_generator, gen1.conv_generator.trainable_variables))

        gen1.loss.append(loss)




        np.savetxt(f"{folder}/PARAMS0_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [gen.params.numpy()], newline='')
        np.savetxt(f"{folder}/PARAMS1_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [gen1.params.numpy()], newline='')
        np.savetxt(f"{folder}/dloss0_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [d_model.loss], newline='')
        np.savetxt(f"{folder}/dloss1_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [d_model1.loss], newline='')
        np.savetxt(f"{folder}/gloss_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", [gen.loss], newline='')
        #if i % 500 == 0:
        #    try: # modified
        #        with open(f"{folder}/ALL_PARAMS_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}", "ab") as f:
        #            np.savetxt(f, [gen.params.numpy()])
        #    except FileNotFoundError:
        #        np.savetxt(f"{folder}/ALL_PARAMS_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}_{lr_d}",  [gen.params.numpy()])
        # serialize weights to HDF5
        #discriminator.save_weights(f"less_image_test/discriminator_4pxls_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")

        # save generators weights

        gen.conv_generator.save_weights(f"{folder}/generator0_4pxls_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")
        gen1.conv_generator.save_weights(f"{folder}/generator1_4pxls_{nqubits}_{latent_dim}_{layers}_{training_samples}_{samples}_{lr}.h5")

    return 0

def build_and_train_model(lr_d=1e-2, lr=5e-1, n_epochs=20000, batch_samples=64, latent_dim=3, layers=3,
                          training_samples=200, pixels=16, nqubits=6, avg=1, dataset=None, folder=None,
                        ):

    # number of qubits generator
    nqubits = nqubits
    # create quantum generator
    gen0 = generator ( latent_dim,
                    layers, nqubits,
                    lr, avg, dataset, 
                    1, pixels,batch_samples ) # digit = 1 because 1's are the inputs 

    gen1 = generator(latent_dim,
                    layers, nqubits,
                    lr, avg, dataset, 
                    0, pixels,batch_samples )

    # create classical discriminator
    disc0 = discriminator( lr_d, pixels)

    disc1 = discriminator( lr_d, pixels)
    
    # train model
    return train(disc0, disc1, gen0, gen1, latent_dim, layers, nqubits, training_samples, 
        n_epochs, batch_samples, lr, lr_d, avg, dataset, folder)

def rebuild_image(img):

    test = np.zeros(shape=(24,24))
    masked =  np.ma.masked_array(test, mask=False)
    masked.mask = generate_condition()
    extracted = np.ma.compressed(masked)
    extracted = img
    masked[~masked.mask] = extracted.ravel()

    return masked.data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default=3, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--training_samples", default=200, type=int)
    parser.add_argument("--n_epochs", default=20000, type=int)
    parser.add_argument("--batch_samples", default=32, type=int)
    parser.add_argument("--pixels", default=64, type=int)
    parser.add_argument("--nqubits", default=6, type=int)
    parser.add_argument("--lr", default=5e-1, type=float)
    parser.add_argument("--lr_d", default=1e-2, type=float)
    parser.add_argument("--dataset", default="8x8Digits", type=str)
    parser.add_argument("--folder", default="cycleGAN", type=str)
    parser.add_argument("--avg", default=32, type=int)
   

    args = vars(parser.parse_args())
    build_and_train_model(**args)