from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam



def model(input_shape):

	X_input = Input(shape = input_shape)

	X = Conv1D(filters = 196, kernel_size = 15, strides = 4)(X_Input)

	X = GRU(units = 128, return_sequences = True)(X)

	X = BatchNormalization()(X)

	X = Activation('relu')(X)

	X = Dropout(rate = .8)(X)

	X = GRU(units = 128, kernel_size = 15, strides = 4)(X)

	X = Dropout(rate = .8)(X)

	X = BatchNormalization()(X)

	X = Dropout(rate = .8)

	X = TimeDistributed(Dense(1, activation = 'sigmoid'))(X)

	model = Model(inputs = X_input, outputs = X)

	return model


def main():
	X_train = np.load("X_Train")
	Y_train = np.load("Y_Train")

	X_Dev = np.load("X_Dev")
	Y_Dev = np.load("Y_Dev")


	model = model(input_shape = (15, 1078))
	model.summary()

	opt = Adam(lr = 0.0001, beta_1 = .9, beta_2 = .999, decay = .01)
	model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ["accuracy"])

	model.fit(X_train, Y_train, batch_size = 5, epochs = 100)

	loss, accuracy = model.evaluate(X_Dev, Y_Dev)

	print("Dev set accuracy:" accuracy)

	model.save_weights("weights.h5")


if __name__ == "__main__":
	main()



