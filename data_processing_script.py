#Terminal Interaction Imports:
import os
import subprocess

#Video and Audio Processing Imports:




#want to create a long list of timestamps
#for each row, it will be considered an indiviual timestamp
#each row will have the concatentated information from the 
#audio input and the visual feature localization
#This information will be written to a txt file
#From there, we can use a seperate script to preprocess the data
#for different LSTM model sizes

#NOTE:
#this implies a certain frame rate. Whatever the frame rate appears to be
#for the video recordings, is what we have to go with for the audio as well
#this also means that when we do live analysis, that the frame rate will
#have to be changed accordingly.
def main():

	video_names = open("video_names.txt", "a")
	audio_names = open("audio_names.txt", "a")
	

	#file_names = []

	safe = True
	video_name = ""
	audio_name = ""
t
	for i in range(191):
		safe = True


		os.chdir("/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/" + str(i + 1))

		if(os.system("find *.avi") == 0):
			video_name = subprocess.check_output("find *.avi", shell = True)#os.system("find *.avi")
			video_name = "/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/" + str(i + 1) + "/" + video_name
		else:
			safe = False

		if os.system("find *.wav") == 0 or safe == False:
			audio_name = subprocess.check_output("find *.wav", shell = True)
			audio_name = "/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/" + str(i + 1) + "/" + audio_name

		else:
			safe = False

		if safe == False:
			video_names.write("error \n")
			audio_names.write("error \n")

		else:
			video_names.write(video_name)
			audio_names.write(audio_name)


if __name__ == "__main__":
	main()