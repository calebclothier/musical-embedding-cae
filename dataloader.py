import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from google.colab import drive
import pandas as pd
import zipfile
import requests
import io
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from urllib.request import urlretrieve
import librosa.display
import librosa
import time
import warnings
from scipy.sparse import coo_matrix
import implicit
from implicit.nearest_neighbours import bm25_weight
import scipy
import pickle


# Mount drive (only works in Google Colab)
drive.mount('/content/drive')


# ================================================
# DOWNLOADING USER-SONG DATA
# ================================================


# Load Echo Nest user taste profile dataset
with zipfile.ZipFile('/content/drive/MyDrive/452_project_data/train_triplets.txt.zip', 'r') as z:
	with z.open('train_triplets.txt') as f:
		df1 = pd.read_csv(f, header=None, delimiter="\t")

df1 = df1.to_numpy()
# Get list of unique songs in dataset
unique_song = np.unique(df1[:,1], return_counts=False)



# ================================================
# COLLABORATIVE FILTERING FOR LATENT SONG FACTORS
# ================================================


df_pandas = pd.DataFrame(df1, columns = ['User','Song','Plays'])
user_rating_matrix = coo_matrix((df_pandas['Plays'].astype(np.int16), (df_pandas['Song'].astype("category").cat.codes.copy(), df_pandas['User'].astype("category").cat.codes.copy())))
# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=64)

# convert to sparse matrix and apply bm25 weighting (calculates confidence from rating)
sparse_user_rating_matrix = user_rating_matrix.tocsr()
sparse_user_rating_matrix = bm25_weight(sparse_user_rating_matrix, K1=100, B=0.8)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(sparse_user_rating_matrix)

# Save the item_factor vectors for every unique song in the dataset
factors = []
all_factors = model.item_factors
i = 0
for song in unique_song:
	factors.append({"MSD_id": song, "item_factor": all_factors[i]})
	i += 1
with open('/content/drive/MyDrive/452_project_data/all_item_factors.pkl', 'wb') as fout:
	pickle.dump(factors, fout)



# ================================================
# DOWNLOADING SONG PREVIEWS FROM SPOTIFY
# ================================================


url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt"
s = requests.get(url).content


def MSD_to_EN_info(msd_id):
	arr = msd_to_en_df.loc[msd_to_en_df['MSD'] == str(msd_id)].iloc[0]
	echonest = arr['EchoNest']
	artist = arr['Artist']
	song = arr['Song']
	return echonest, song, artist

def lookup_spotify(song, artist):
	sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f4f7d65dd3ec4af4a6e437a5acdc8cbb",
															  client_secret="bf7224932a1149ea9bd043056f7b6677"))
	query = "artist: %s track: %s" % (artist, song)
	return sp.search(q=query, type="track")

def get_track_preview_url(sp_uri):
	sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f4f7d65dd3ec4af4a6e437a5acdc8cbb",
															 client_secret="bf7224932a1149ea9bd043056f7b6677"))
	results = sp.artist_top_tracks(sp_uri)
	return results['tracks'][0]['preview_url']

def get_mel_spectrogram(preview_url, plot=False):

	local_filename, headers = urlretrieve(preview_url)
	x, sr = librosa.load(local_filename)

	if sr != 22050:
		print("Error: sampling rate not 22050")
		print(sr)

	# Create spectrogram
	X = librosa.feature.melspectrogram(x, sr=sr)
	# Amplitude to db
	Xdb = librosa.amplitude_to_db(X, ref=np.min)

	if plot:
		plt.figure(figsize=(14, 5))
		librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='mel') 
		plt.colorbar()

	return Xdb, sr

def songid_to_spectrogram(song_id, max_tries=5):
	_, track, artist = MSD_to_EN_info(song_id)
	for i in range(max_tries):
		try:
			res = lookup_spotify(track, artist)
			break
		except Exception:
			time.sleep(5) 
			continue
	items = res['tracks']['items']
	if len(items) > 0:
		# Check if there's a preview for the song in the Spotify database
		if (items[0]['preview_url']):
			preview_url = items[0]['preview_url']
			Xdb, sr = get_mel_spectrogram(preview_url)
			return {'MSD_id': song, 'Song': track, 'Artist': artist, 'spectrogram': Xdb, 'sr': sr}
		else:
			return None
	else:
		return None


# Loop for downloading and saving Spotify previews
total = 0
run = 0
num_songs = 5000
spectrograms = []

for song in unique_song[15200:]:
  if song in loaded_ids:
    continue
  else:
    try:
      data = songid_to_spectrogram(str(song))
    except:
      total += 1
      continue
    if data:
        run += 1
        spectrograms.append(data)
    total += 1
    if ((total % 100) == 0):
      print("Song %d, %f" % (total, run/total))
      print(len(spectrograms))
    if (run == num_songs):
      break

print("Fraction of songs with Spotify previews = %f" % (run / total))

# Save spectrograms
with open('/content/drive/MyDrive/452_project_data/5k_spectrograms_3_dbcorrected.pkl', 'wb') as fout:
    pickle.dump(spectrograms, fout)




