
import numpy as np
import cv2
import dlib
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt

#NOTES:
#This file is meant just ot interact and see what the data looks like, what form it comes in...
#This also allows you to visual which facial landmarks are actually being tracked
 
 
cap = cv2.VideoCapture('/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/4/S001-005.avi')
#cap.set(cv2.CV_CAP_PROP_FPS, 10)
#cap.set(cv2.cv.CV_CAP_PROP_FPS, 10) 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#audio = AudioSegment.from_file()
#_, data = wavfile.read('/Users/ethanhellman/Desktop/CS230/FinalProject/MAHNOB Files/mahnob-laughter-database_download_2020-05-22_20_39_52/Sessions/75/S009-004_mic.wav')

#print(np.shape(data))
#print(len(data))
#print(data[:,0].shape)


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 1884 # Length of each window segment
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
    rate, data = wavfile.read(wav_file)
    return rate, data


x = graph_spectrogram('/Users/ethanhellman/Desktop/CS230/FinalProject/face/Sessions/4/S001-005_mic.wav')
x  = x.swapaxes(0,1)
print(np.shape(x))

#for i in range(len(x)):
#	print(x[i, :])


total_frames = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
    	break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        #x1 = face.left()
        #y1 = face.top()
        #x2 = face.right()
        #y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


    #plt.figure()
    #plt.imshow(frame)te

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #fps = cap.get(5)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.imshow('frame', frame)

    #print(cap.CV_CAP_PROP_FPS)

    #print(fps)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

 
    #if cv2.waitKey(1) &amp; 0xFF == ord('q'):
    #    break
    total_frames += 1
 
 
print(total_frames)

cap.release()
cv2.destroyAllWindows()