from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.animation import Animation
import cv2, sys, numpy, os
from PIL import Image
from gtts import gTTS
import pytesseract
import time
def animate(instance):
        # create an animation object. This object could be stored
        # and reused each call or reused across different widgets.
        # += is a sequential step, while &= is in parallel
        animation = Animation(pos=(100, 100), t='out_bounce')
        animation += Animation(pos=(200, 100), t='out_bounce')
        animation &= Animation(size=(500, 500))
        animation += Animation(size=(100, 50))

        # apply the animation on the button, passed in the "instance" argument
        # Notice that default 'click' animation (changing the button
        # color while the mouse is down) is unchanged.
        animation.start(instance)
def callback6(instance):
        #video=cv2.VideoCapture(0)
        #check,frame=video.read()
        #time.sleep(2)
        #cv2.imwrite("text_img.jpg",frame)
        #cv2.waitKey(0)
        #video.release()
        pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        txt=pytesseract.image_to_string(Image.open("C:\\Users\\DELL\\Pyproject\\text_img.png"),lang="eng")
        print(txt)
        speech=gTTS(txt)
        speech.save("1.mp3")
        cv2.destroyAllWindows()
def callback1(instance):
		# Creating database 
	# It captures images and stores them in datasets 
	# folder under the folder name of sub_data 
	#haar_file ='C:\Users\DELL\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_defalt.xml'
	haar_file="C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
	# All the faces data will be 
	# present this folder 
	datasets = 'datasets'


	# These are sub data sets of folder, 
	# for my faces I've used my name you can 
	# change the label here 
	sub_data = 'vivek'	

	path = os.path.join(datasets, sub_data) 
	if not os.path.isdir(path): 
		os.mkdir(path) 

	# defining the size of images 
	(width, height) = (130, 100)	 

	#'0' is used for my webcam, 
	# if you've any other camera 
	# attached use '1' like this 
	face_cascade = cv2.CascadeClassifier(haar_file) 
	webcam = cv2.VideoCapture(0) 

	# The program loops until it has 30 images of the face. 
	count = 1
	while count < 30: 
		(_, im) = webcam.read() 
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
		faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
		for (x, y, w, h) in faces: 
			cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
			face = gray[y:y + h, x:x + w] 
			face_resize = cv2.resize(face, (width, height)) 
			cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
		count += 1
		
		cv2.imshow('OpenCV', im) 
		key = cv2.waitKey(10) 
		if key == 27: 
			break
def callback2(instance):
  
        haar_file ="C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
        datasets = 'datasets'

        # Part 1: Create fisherRecognizer 
        print('Recognizing Face Please Be in sufficient Lights...') 

        # Create a list of images and a list of corresponding names 
        (images, lables, names, id) = ([], [], {}, 0) 
        for (subdirs, dirs, files) in os.walk(datasets): 
                for subdir in dirs: 
                        names[id] = subdir 
                        subjectpath = os.path.join(datasets, subdir) 
                        for filename in os.listdir(subjectpath): 
                                path = subjectpath + '/' + filename 
                                lable = id
                                images.append(cv2.imread(path, 0)) 
                                lables.append(int(lable)) 
                        id += 1
        (width, height) = (130, 100) 

        # Create a Numpy array from the two lists above 
        (images, lables) = [numpy.array(lis) for lis in [images, lables]] 

        # OpenCV trains a model from the images 
        # NOTE FOR OpenCV2: remove '.face' 
        model = cv2.face.LBPHFaceRecognizer_create() 
        model.train(images, lables) 

        # Part 2: Use fisherRecognizer on camera stream 
        face_cascade = cv2.CascadeClassifier(haar_file) 
        webcam = cv2.VideoCapture(0) 
        while True: 
                (_, im) = webcam.read() 
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
                faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
                for (x, y, w, h) in faces: 
                        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                        face = gray[y:y + h, x:x + w] 
                        face_resize = cv2.resize(face, (width, height)) 
                        # Try to recognize the face 
                        prediction = model.predict(face_resize) 
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 

                        if prediction[1]<80: 
                                cv2.putText(im, '% s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
                        else: 
                                cv2.putText(im, 'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 

                cv2.imshow('OpenCV', im) 
                
                key = cv2.waitKey(10) 
                if key == 27: 
                        break
def callback3(instance):
        cap = cv2.VideoCapture(0)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                #frame = cv2.flip(frame,0)

                # write the flipped frame
                out.write(frame)

                cv2.imshow('frame',frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
def callback4(instance):
        cap = cv2.VideoCapture('output.avi')
        while(cap.isOpened()):
            ret, frame = cap.read()

            cv2.imshow('frame',frame)
            cv2.waitKey(30)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
class MyGrid(GridLayout):
	def __init__(self,**kwargs):
		super(MyGrid,self).__init__(**kwargs)
		self.btn1 = Button(text='Take Samples',font_size=14,background_normal='',background_color=[255, 0, 0, 1],pos=(200,200))
		self.btn2 = Button(text='Run Test',font_size=14,background_normal='',background_color=[255, 0, 0, 1],pos=(400,200))
		self.add_widget(self.btn1)
		self.btn1.bind(on_press=callback1)
		self.btn5 = Button(text='Run Video',font_size=14,background_normal='',background_color=[255, 0, 0, 1],pos=(200,400))
		self.add_widget(self.btn2)
		self.btn2.bind(on_press=callback2)
		self.btn3 = Button(text='Record Video',font_size=14,background_normal='',background_color=[255, 0, 0, 1],pos=(400,400))
		self.add_widget(self.btn3)
		self.btn3.bind(on_press=callback3)
		self.add_widget(self.btn5)
		self.btn5.bind(on_press=callback4)
		self.btn6 = Button(text='Text To Speech',font_size=14,background_normal='',background_color=[255, 0, 0, 1],pos=(600,400))
		self.btn6.bind(on_press=callback6)
		self.add_widget(self.btn6)
		

class Simple(App):
	def build(self):
		return MyGrid()

			
if __name__=="__main__":
	Simple().run()
	
