import numpy as np
import csv
import cv2
import dlib
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as special_shuffle



# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    if rate == False:
    	return False
    nfft = 1883 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
	try:
		rate, data = wavfile.read(wav_file)
		return rate, data
	except IOError:
		print("File could not be opened")
		return False, 0

	

def main():
	audio_files = open("audio_names.txt", "r")
	audio_names = audio_files.readlines()

	video_files = open("video_names.txt", "r")
	video_names = video_files.readlines()

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	print("Hello")

	dataset_x = np.zeros((500000, 15, 1078))
	dataset_y = np.zeros((500000, 15, 1))

	#np.save("X_temp", dataset_x)
	#np.save("Y_temp", dataset_y)

	num_data_points = 0

	#cap = cv2.VideoCapture('/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/1/S001-001.avi')

	#while(cap.isOpened()):
	#	rate, frame = cap.read()

	#	if rate:
	#		cv2.imshow("frame", frame)
	#cap.release()

	#facial_features = []

	sanity = 0

	for i in range(len(audio_names)):
		print("Running")
		sanity = 0
		

		if audio_names[i].strip() == "error":
			continue
		else:

			#Audio Processing:	
			print(audio_names[i].strip())
			
			audio_features = graph_spectrogram(str(audio_names[i].strip()))
			if audio_features.all() == False: continue
			audio_features  = audio_features.swapaxes(0,1)
			#print(np.shape(audio_features))

			#Video Processing:
			facial_features_temp = []

			print(video_names[i].strip())

			try:
				cap = cv2.VideoCapture(str(video_names[i].strip()))
			except IOError:
				print("File could not be opened")
				continue


			#dataset_x = np.load("X_temp")
			#dataset_y = np.load("Y_temp")

			while(cap.isOpened()):
				sanity += 1
				ret, frame = cap.read()
				features = []
				if ret == True:
					gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					faces = detector(gray)

					for face in faces:
						landmarks = predictor(gray, face)

						for n in range(0, 68):
						    x = landmarks.part(n).x
						    y = landmarks.part(n).y
						    features.append(x)
						    features.append(y)
						    #cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

					#print(str(features[0]) + ", " + str(features[1]))

					#cv2.imshow('frame', frame)

					facial_features_temp.append(features)

				else:
					break
			fps = cap.get(cv2.CAP_PROP_FPS)
			print(fps)
			cap.release()
			#facial_features_temp = facial_features_temp[np.abs(len(facial_features_temp) - np.shape(audio_features)[0]):]
			#print(str(len(facial_features)) + ", " +str(len(facial_features[0])))
			print(sanity)
			facial_features = np.array(facial_features_temp)
			print(np.shape(facial_features))

			if np.shape(audio_features)[0] < np.shape(facial_features)[0]:
				continue

			#Reshape audio in case it is larger than facial:
			audio_features = audio_features[:np.shape(facial_features)[0],:]
			print(np.shape(audio_features))


			#CSV, Y annotations:
			annotations = np.zeros((np.shape(facial_features)[0], 1))
			annotation_file = open("/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/" + str(i+1) + "/laughterAnnotation.csv")
			csvreader = csv.reader(annotation_file)

			fields = next(csvreader)

			for row in csvreader:
				if row[1] == "Laughter" or row[1] == "PosedLaughter" or row[1] == "SpeechLaughter" :
					start_index = int(round(fps*float(row[2])))
					end_index = int(round(fps*float(row[3])))
					annotations[start_index:end_index, :] = 1
			annotations = annotations[:np.shape(facial_features)[0], :]
			print(np.shape(annotations))
			#for j in range(np.shape(annotations)[0]):
			#	print(annotations[j, :])


			#NOW turn them into data points -- concat the two np arrays, then create a bunch of mini data points
			#which will be sequences of frames. The sequences should be 

			#First lets run a check on the dimensions:
			if len(np.shape(facial_features)) != 2 or np.shape(facial_features)[1] != 136:
				print("Error with shape")
				continue
			elif len(np.shape(audio_features)) != 2 or np.shape(audio_features)[1] != 942:
				print("Error with shape")
				continue
			elif len(np.shape(annotations)) != 2 or np.shape(annotations)[1] != 1:
				print("Error with shape")
				continue


			joined = np.concatenate((facial_features, audio_features), axis = 1)
			print(np.shape(joined))

			for j in range(np.shape(joined)[0]):
				sample_x = np.zeros((15, 1078))
				temp_x = np.zeros((15, 1078))

				sample_y = np.zeros((15, 1))
				temp_y = np.zeros((15, 1)) 

				if j < 15:
					sample_x[(15 - j - 1):, :] = joined[:j+1, :]
					sample_y[(15 - j - 1):, :] = annotations[:j+1, :]

				elif j > np.shape(joined)[0] - 15:
					sample_x[:(np.shape(joined)[0] - j), :] = joined[j:, :]
					sample_y[:(np.shape(joined)[0] - j), :] = annotations[j:, :]

				else:
					sample_x[:,:] = joined[j - 15: j, :] 
					sample_y[:,:] = annotations[j - 15: j, :]


				#Add newly minted data point to the list of all of the datapoints	
				dataset_x[num_data_points, :, :] = sample_x
				dataset_y[num_data_points, :, :] = sample_y


				#Make sure that we are keeping track of our total number of data points
				num_data_points += 1


			print(num_data_points)

			np.save("X_temp", dataset_x[:num_data_points, :, :])
			np.save("Y_temp", dataset_y[:num_data_points, :, :])

			print("Saved data from file #" + str(i))


			#Make sure that we are keeping track of our total number of data points
			#num_data_points += np.shape(joined)[0] + 28

			#Now turn them into sequences

	#Make sure that we have the right number of data points:
	dataset_x = dataset_x[:num_data_points, :, :]
	dataset_y = dataset_y[:num_data_points, :, :]


	#shuffle the datasets and split them into train, dev, test:
	shuffled_x, shuffled_y = special_shuffle(dataset_x, dataset_y, random_state = 0)

	#We are going to do a 70/15/15 split
	trains = int(round(.7*num_data_points))
	dev = int(round(.15*num_data_points))

	x_train = shuffled_x[:trains, :, :]
	y_train = shuffled_y[:trains, :, :]

	x_dev = shuffled_x[trains:(trains + dev), :, :] 
	y_dev = shuffled_y[trains:(trains + dev), :, :] 

	x_test = shuffled_x[(trains + dev):, :, :]
	y_test = shuffled_y[(trains + dev):, :, :]


	#Now save the data sets
	np.save("X_Train", x_train)
	np.save("Y_Train", y_train)

	np.save("X_Dev", x_dev)
	np.save("Y_Dev", y_dev)

	np.save("X_Test", x_test)
	np.save("Y_Test", y_test)




if __name__ == "__main__":
	main()