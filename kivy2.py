import kivy
kivy.require("1.9.0")
import cv2, sys, numpy, os
from PIL import Image
from gtts import gTTS
import time
import pytesseract
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.popup import Popup
 
# Used to display popup
class CustomPopup1(Popup):
    pass
class CustomPopup2(Popup):
    pass
class SampBoxLayout(BoxLayout):
    # For checkbox
    checkbox_is_active = ObjectProperty(False)
    def checkbox_18_clicked(self, instance, value):
        if value is True:
            print("Checkbox Checked")
        else:
            print("Checkbox Unchecked")
 
    # For radio buttons
    blue = ObjectProperty(True)
    red = ObjectProperty(False)
    green = ObjectProperty(False)
    def callback1(instance):
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
    # For Switch
    def switch_on(self, instance, value):
        if value is True:
            print("Switch On")
        else:
            print("Switch Off")
 
    # Opens Popup when called
    def callback2(instance):
          
        #haar_file ='C:\Users\DELL\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_defalt.xml'
        haar_file="C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
        # All the faces data will be 
        # present this folder 
        datasets = 'datasets'


       
        sub_data = 'vivek'  

        path = os.path.join(datasets, sub_data) 
        if not os.path.isdir(path): 
            os.mkdir(path) 

        # defining the size of images 
        (width, height) = (130, 100)     

     
        face_cascade = cv2.CascadeClassifier(haar_file) 
        webcam = cv2.VideoCapture(0) 

        
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
    # For Spinner
    def callback3(instance):
  
        haar_file ="C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
        datasets = 'datasets'

        
        print('Recognizing Face Please Be in sufficient Lights...') 

       
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

        
        (images, lables) = [numpy.array(lis) for lis in [images, lables]] 

       
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
    def callback4(instance):
        cap = cv2.VideoCapture(0)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
             
                out.write(frame)

                cv2.imshow('frame',frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    def callback5(instance):
        cap = cv2.VideoCapture('output.avi')
        while(cap.isOpened()):
            ret, frame = cap.read()

            cv2.imshow('frame',frame)
            cv2.waitKey(30)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
class SampleApp(App):
    def build(self):
 
        # Set the background color for the window
        Window.clearcolor = (1, 1, 1, 1)
        return SampBoxLayout()
if __name__ == '__main__':
    sample_app = SampleApp()
    sample_app.run()
 
# ---------- sample.kv  ----------
 

 
