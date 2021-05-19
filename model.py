import pickle
import numpy as np
from matplotlib import pyplot as plt
from google.colab import drive
import librosa.display
import librosa
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# ================================================
# DEFINE CONVOLUTIONAL AUTOENCODER MODEL
# ================================================

class ConvolutionalAutoencoder(tf.keras.Model):

  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    # Encoder
    self.conv_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(256, 128), kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='conv_1')
    self.conv_2 = tf.keras.layers.Conv1D(256, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='conv_2')
    self.conv_3 = tf.keras.layers.Conv1D(512, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='conv_3')

    self.pool_1 = tf.keras.layers.MaxPool1D(pool_size=2, name='pool_1')
    self.pool_2 = tf.keras.layers.MaxPool1D(pool_size=2, name='pool_2')
    self.pool_3 = tf.keras.layers.MaxPool1D(pool_size=2, name='pool_3')

    self.dense_1 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.01),  name='dense_1')
    self.dense_2 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='dense_2')
    self.dense_3 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='dense_3')
    self.embedding_layer = tf.keras.layers.Dense(256, activation='linear', name='embedding')

    # Decoder
    self.dense_4 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name="dense_4")
    self.dense_5 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name="dense_5")
    self.dense_6 = tf.keras.layers.Dense(32*512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='dense_6')

    self.deconv_1 = tf.keras.layers.Conv1DTranspose(filters=512, kernel_size=2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='deconv_1')
    self.deconv_2 = tf.keras.layers.Conv1DTranspose(256, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='deconv_2')
    self.deconv_3 = tf.keras.layers.Conv1DTranspose(128, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(l=0.01), name='deconv_3')

    self.upsample_1 = tf.keras.layers.UpSampling1D(size=2, name='upsample_1')
    self.upsample_2 = tf.keras.layers.UpSampling1D(size=2, name='upsample_2')
    self.upsample_3 = tf.keras.layers.UpSampling1D(size=2, name='upsample_3')

  def call(self, x):
    # Encode
    x = self.conv_1(x)
    x = self.pool_1(x)
    x = self.conv_2(x)
    x = self.pool_2(x)
    x = self.conv_3(x)
    x = self.pool_3(x)
    x = tf.reshape(x, (-1, 32*512))
    x = self.dense_1(x)
    x = self.dense_2(x)
    embedding = self.embedding_layer(x)
    # Decode
    x = self.dense_4(embedding)
    x = self.dense_5(x)
    x = self.dense_6(x)
    x = tf.reshape(x, [-1, 32, 512])
    x = self.upsample_1(x)
    x = self.deconv_1(x)
    x = self.upsample_2(x)
    x = self.deconv_2(x)
    x = self.upsample_3(x)
    x = self.deconv_3(x)
    return x, embedding

  def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            x_recon, embedding = self(x)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(x, x_recon)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, x_recon)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

  def model(self):
        x = tf.keras.layers.Input(shape=(256, 128))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))




# ================================================
# TRAINING LOOP
# ================================================
def main():

	# Define sampling rate
	sr = 22050


	# Mount drive (only works in Google Colab)
	drive.mount('/content/drive')


	# ================================================
	# LOAD TRAINING DATA
	# ================================================

	# Load pickle files
	with open('/content/drive/MyDrive/452_project_data/5k_spectrograms_dbcorrected.pkl', 'rb') as fin:
	    spectrograms = pickle.load(fin)
	with open('/content/drive/MyDrive/452_project_data/5k_spectrograms_2_dbcorrected.pkl', 'rb') as fin:
	    spectrograms2 = pickle.load(fin)

	# Create Tensorflow datasets
	train_data = tf.data.Dataset.from_tensor_slices([tf.transpose(tf.convert_to_tensor(spec['spectrogram'][:, 256*i:256*(i+1)], dtype=tf.float32)) for i in range(4) for spec in (spectrograms + spectrograms2[:4000]) if (spec['spectrogram'].shape[1] >= 1024)])
	test_data = tf.data.Dataset.from_tensor_slices([tf.transpose(tf.convert_to_tensor(spec['spectrogram'][:, 256*i:256*(i+1)], dtype=tf.float32)) for i in range(4) for spec in spectrograms2[4000:] if (spec['spectrogram'].shape[1] >= 1024)])

	train_dcomb = tf.data.Dataset.zip((input_data, input_data))
	train_dcomb = train_dcomb.batch(batch_size=32, drop_remainder=True)
	test_dcomb = tf.data.Dataset.zip((test_data, test_data))

	# ================================================
	# TRAIN AND SAVE MODEL
	# ================================================
	cae_model = ConvolutionalAutoencoder(name='cae1')
	cae_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002), 
	                  loss=tf.keras.losses.MeanSquaredError(),
	                  metrics=['MeanSquaredError'])
	print(cae_model.model().summary())
	cae_model.fit(x=dcomb, epochs=50, shuffle=True, batch_size=32)
	cae_model.save('/content/drive/MyDrive/452_project_data/cae_50epochs')



# ================================================
# FUNCTIONS FOR EVALUATING RESULTS
# ================================================

def get_embeddings(dcomb):
	i = 0
	for elem in dcomb.take(1249):
		res, embeddings = cae_model.predict(elem[0])
		if (i == 0):
	    	all_embeddings = embeddings
	  	else:
	    	all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)
	  	i += 1
	 return all_embeddings

def PCA_plot(embeddings, plot=True):
	# Returns the first 2 principal components of every song embedding
	# Optionally plots these principal components
	pca = PCA(n_components=2)
	pca.fit(embeddings)
	embeddings_pca = pca.transform(embeddings)
	plt.figure(figsize=(8,8))
	if plot:
		plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], s=1)
		plt.show()
	return embeddings_pca

def get_cosine_sim_matrix(embeddings_pca):
	cos_sim = cosine_similarity(embeddings_pca)
	return cos_sim


if __name__ == "__main__":
	main()





