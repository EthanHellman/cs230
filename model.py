import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


########
#This file creates and trains an instance of the model.
#Saves weights for later use for another model/for a different framework such 
#as real-time detection
########


def create_model(input_shape):

	X_input = Input(shape = input_shape)

	X = Conv1D(filters = 196, kernel_size = 15, strides = 4)(X_input)

	X = GRU(units = 128, return_sequences = True)(X)

	X = BatchNormalization()(X)

	X = Activation('relu')(X)

	X = Dropout(rate = .2)(X)

	X = GRU(units = 128, return_sequences = True)(X)

	X = Dropout(rate = .2)(X)

	X = BatchNormalization()(X)

	X = Dropout(rate = .2)(X)

	X = TimeDistributed(Dense(15, activation = 'sigmoid'))(X)

	model = Model(inputs = X_input, outputs = X)

	return model


def main():
	X_train = np.load("X_Train.npy")
	Y_train = np.load("Y_Train.npy")

	X_Dev = np.load("X_Dev.npy")
	Y_Dev = np.load("Y_Dev.npy")

	Y_train = np.reshape(Y_train, (np.shape(Y_train)[0], 1, 15))

	Y_Dev = np.reshape(Y_Dev, (np.shape(Y_Dev)[0], 1, 15))

	#model_checkpoint_callback = ModelCheckpoint("/checkpoint.h5", save_weights_only = True, monitor = "", mode = "max", save_best_only = True)


	model = create_model(input_shape = (15, 1078))

	model.summary()

	opt = Adam(lr = 0.001, beta_1 = .9, beta_2 = .999, decay = .01)
	model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ["accuracy"])

	#model.fit(X_train, Y_train, batch_size = 10, epochs = 100, callbacks = [model_checkpoint_callback])

	model.fit(X_train, Y_train, batch_size = 10, epochs = 100)

	loss, accuracy = model.evaluate(X_Dev, Y_Dev)

	print("Dev set accuracy:" + str(accuracy))

	model.save_weights("weights.h5")


if __name__ == "__main__":
	main()



