import numpy as np
import csv
import cv2
import dlib
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as special_shuffle



def main():

	#GOAL: shuffle the datasets and split them into train, dev, test:
	dataset_x = np.load("X_temp.npy")
	dataset_y = np.load("Y_temp.npy")

	print(np.shape(dataset_x))
	print(np.shape(dataset_y))


	#Want to seperate out the "positive" examples from the "negative" examples to 
	#achieve a better distribution of data
	y_summed = np.sum(dataset_y, axis = 1)
	x_positive = dataset_x[np.where(y_summed[:, 0] >= 5)]
	y_positive = dataset_y[np.where(y_summed[:, 0] >= 5)]

	#test = np.sum(dataset_y, axis = 1)
	#test_1 = dataset_x[np.where(test[:, 0] >= 5)]
	#print("Shape of test:")
	#print(np.shape(test_1))

	#x_positive = dataset_x[np.sum(dataset_y, axis = 1) >= 5]
	#y_positive = dataset_y[np.sum(dataset_y, axis = 1) >= 5]

	print("Positive Examples:")
	print(np.shape(x_positive))
	print(np.shape(y_positive))

	x_negative = dataset_x[np.where(y_summed[:, 0] < 5)]
	y_negative = dataset_y[np.where(y_summed[:, 0] < 5)]

	#x_negative = dataset_x[np.sum(dataset_y, axis = 1) < 5]
	#y_negative = dataset_y[np.sum(dataset_y, axis = 1) < 5]

	print("Negative Examples:")
	print(np.shape(x_negative))
	print(np.shape(y_negative))

	#Get a 50/50 distribution:
	x_negative_shuffled, y_negative_shuffled = special_shuffle(x_negative, y_negative, random_state = 0)

	x_negative_chosen = x_negative_shuffled[:np.shape(x_positive)[0], :, :]
	y_negative_chosen = y_negative_shuffled[:np.shape(x_positive)[0], :, :]

	#Now that we have a 50/50 distribution, combine them again, and reshuffle
	#before splitting into train, dev, test:
	x_joined = np.concatenate((x_positive, x_negative_chosen), axis = 0)
	y_joined = np.concatenate((y_positive, y_negative_chosen), axis = 0)

	x_joined_shuffled, y_joined_shuffled = special_shuffle(x_joined, y_joined, random_state = 0)

	print("Joined and Shuffled Examples:")
	print(np.shape(x_joined_shuffled))
	print(np.shape(y_joined_shuffled))



	#x_positive_shuffled, y_positive_shuffled = special_shuffle(x_positive, y_positive, random_state = 0)
	

	#shuffled_x, shuffled_y = special_shuffle(dataset_x, dataset_y, random_state = 0)

	#We are going to do a 70/15/15 split
	#trains = int(round(.7*np.shape(dataset_x)[0]))
	#dev = int(round(.15*np.shape(dataset_x)[0]))

	#x_train = shuffled_x[:trains, :, :]
	#y_train = shuffled_y[:trains, :, :]

	#x_dev = shuffled_x[trains:(trains + dev), :, :] 
	#y_dev = shuffled_y[trains:(trains + dev), :, :] 

	#x_test = shuffled_x[(trains + dev):, :, :]
	#y_test = shuffled_y[(trains + dev):, :, :]



	trains = int(round(.7*np.shape(x_joined_shuffled)[0]))
	dev = int(round(.15*np.shape(x_joined_shuffled)[0]))

	x_train = x_joined_shuffled[:trains, :, :]
	y_train = y_joined_shuffled[:trains, :, :]

	x_dev = x_joined_shuffled[trains:(trains + dev), :, :] 
	y_dev = y_joined_shuffled[trains:(trains + dev), :, :] 

	x_test = x_joined_shuffled[(trains + dev):, :, :]
	y_test = y_joined_shuffled[(trains + dev):, :, :]



	print("Original Dataset Dimensions:")
	print(np.shape(dataset_x))
	print(np.shape(dataset_y))

	print("Training Set Dimensions:")
	print(np.shape(x_train))
	print(np.shape(y_train))

	print("Dev Set Dimensions:")
	print(np.shape(x_dev))
	print(np.shape(y_dev))

	print("Test Set Dimensions:")
	print(np.shape(x_test))
	print(np.shape(y_test))


	#Now save the data sets
	np.save("X_Train", x_train)
	np.save("Y_Train", y_train)

	np.save("X_Dev", x_dev)
	np.save("Y_Dev", y_dev)

	np.save("X_Test", x_test)
	np.save("Y_Test", y_test)


if __name__ == "__main__":
	main()