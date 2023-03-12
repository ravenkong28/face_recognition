#2301865602 - Gaudentius Matthew Setiadi
#2301923294 - Raven Kongnando Lasher

import os
import cv2
import numpy as np

classifier = cv2.CascadeClassifier("../Project_lab/haarcascades/haarcascade_frontalface_default.xml")


def get_path_list(root_path):
    train_names = list()
    for folder in os.listdir(root_path):
        train_names.append(folder)
    # print(train_names) --> Chris Hemsworth,  Elizabeth Olsen, dll
    return train_names
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

def get_class_id(root_path, train_names):
    image_classes_list = list()
    train_image_list = list()    
    # total=0
    for index, class_path in enumerate(train_names):
        # print(index, path) -> 0 Chris Hemsworth
        train_path_list = os.listdir(root_path +'/'+class_path)
        #print(train_path_list) #Nama foto file masing" per list
        for img in train_path_list:
            train_image_path = f"{root_path}/{class_path}/{img}"
            train_image_list.append(train_image_path)
            image_classes_list.append(index)
            # total+=1
    
    # print("Total foto",total)
    # print("Train Image Path = ")
    # print(train_image_list)
    #print(image_classes_list)

    return train_image_list, image_classes_list
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''

def detect_faces_and_filter(image_list, image_classes_list=None):
    train_face_grays = list()
    filtered_location = list()
    filtered_classes_list = list()
    if image_classes_list == None :
        image_classes_list = np.zeros(len(image_list))

    for idx, image in zip(image_classes_list,image_list) :
        image_read = cv2.imread(image,0)
        
        detected_faces = classifier.detectMultiScale(image_read, scaleFactor = 1.3, minNeighbors = 5)
        
        if len(detected_faces) < 1:
            continue
        for face in detected_faces :
            x, y, h, w = face
            
            face_images = image_read[y:y+h, x:x+w]
            train_face_grays.append(face_images)
            filtered_location.append(face)
            filtered_classes_list.append(idx)

    return train_face_grays, filtered_location, filtered_classes_list
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path):
    test_image_list = list()
    for img in (os.listdir(test_root_path)):
        test_image_list.append(f"{test_root_path}/{img}")
    
    return test_image_list
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    
def predict(recognizer, test_faces_gray):
    predict_results = list() #1
    for image in test_faces_gray:
        result, percentage= recognizer.predict(image)
        predict_results.append(result)
        #print(test_faces_gray)
        # print(predict_results)
    return predict_results
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    predicted_test_image_list = list()
    for idx, (img, face) in enumerate(zip(test_image_list, test_faces_rects)):
        x,y,h,w = face
        image_read= cv2.imread(img)
        cv2.rectangle(image_read, (x,y), (x+w,y+h), (0,255,0),2)
        # j = 0
        # for i in predict_results:
        #     text = train_names[idx]
        #     print("I, J dan Text=", i, text)
        #     j+=1
        #     break
        text = train_names[predict_results[idx]]
        cv2.putText(image_read, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN,4,(0,0,255), 4)
        test_image = cv2.resize(image_read,(350,350)) 
        
        predicted_test_image_list.append(test_image)
    return predicted_test_image_list

    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    

def combine_and_show_result(image_list):
    output = list()
    for image in image_list :
        output.append(image)
        # cv2.imshow("Final Result", image)
        # cv2.waitKey(0)
    
    # output_image = np.hstack ((output[0],output[1],output[2],output[3],output[4]))
    cv2.imshow("Final Result", cv2.hconcat(output))
    cv2.waitKey(0)
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    train_root_path = '../Project_lab/dataset/train'
    
    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)
    
    test_root_path = '../Project_lab/dataset/test'
    
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    print(predict_results)
    combine_and_show_result(predicted_test_image_list)
