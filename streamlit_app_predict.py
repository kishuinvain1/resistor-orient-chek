import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64





def load_image():
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image)
       
    return path, opencv_image
       


	
def loadSegFormModel():
    print("...loading...segformer..")
    rf = Roboflow(api_key="uhDFc9G6MKjrEvbfHt6B")
    project = rf.workspace().project("fleetguardcrack")
    #model = project.version(1).model
    model = project.version(4).model
    return model

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return np.array(image)
	
def segFormCrack(cl, x, y, w, h, cnf, saved_image, bias):

    print(".....inside segFormCrack......")
    img = cv2.imread(saved_image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #print(img.shape)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    bias = int(bias)
    roi = img[y-h//2+bias:y+h//2-bias, x-w//2+bias:x+w//2-bias, :]
    st.image(roi, caption="ROI")
    cv2.imwrite("saved_ROI.jpg", roi)
    segform_model = loadSegFormModel()
    preds = segform_model.predict("saved_ROI.jpg")
    print("segmentation results are ")
    seg_mask = preds[0]['segmentation_mask']
	
    
	
    print(seg_mask)
    #seg_mask_read = cv2.imread(seg_mask, 0)
    #cv2.imwrite("seg_mask.jpg", seg_mask_read)
    #seg_img = Image.open("seg_mask.jpg")
    im_bytes = base64.b64decode(seg_mask)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    
    seg_mask_image = cv2.imdecode(im_arr, 0)
    print(seg_mask_image)
    nz_cmp = np.sum(seg_mask_image)
    print("non-zero is......................................")
    print(nz_cmp)
    
        

    #pil_image = stringToImage(seg_mask)
    #seg_mask_image = toRGB(pil_image)
    
    #st.image(seg_mask_image, caption='segmentation mask')
    
    
    if(nz_cmp > 10):
        preds = segform_model.predict("saved_ROI.jpg").save("crack_pred.jpg")
        crck_pred = Image.open('crack_pred.jpg')
        st.image(crck_pred, caption='crack localization')
    else:
        st.write("No Crack Detected")
	
def drawBoundingBox(results, saved_image):
    #img = Image.open(saved_image)
    

    img = cv2.imread(saved_image)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for item in results:
        x = int(item['x'])
        y = int(item['y'])
        w = int(item['width'])
        h = int(item['height'])
        cl = item['class'] 
        start_pnt = (x-w//2,y-h//2)
        end_pnt = (x+w//2, y+h//2)
        txt_start_pnt = (x-w//2, y-h//2-15)

        if(cl == "Ok"):
            img = cv2.rectangle(img, start_pnt, end_pnt, (0,255,0), 10)
            #img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10, cv2.LINE_AA)
        elif(cl == "Not-Ok"):
            img = cv2.rectangle(img, start_pnt, end_pnt, (0,0,255), 10)
            #img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10, cv2.LINE_AA)
		
    cv2.imwrite("dets.jpg", img)
    res_img = Image.open("dets.jpg")
    st.image(res_img, caption="Detection Results")

    
	
    
    


def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	

	
def dispResults(results):
       

    cv2.imwrite("dets.jpg", img)
            
	
	
def main():
    st.title('Resistor Orient Check') 
    image, svd_img = load_image()
    zoomin_bias = st.number_input('Zoomin Bias')
   
    result = st.button('Predict')
    if(result):
        st.write('Calculating results...')
        
	#Model trained on 08/04/23
        rf = Roboflow(api_key="q3ZrI4IarL2A3pHuOrt2")
        project = rf.workspace().project("resistor-orient-check")
        model = project.version(2).model
	
	
	
	
        #results = model.predict("main_image.jpg", confidence=40, overlap=30)
        results = model.predict(svd_img, confidence=40, overlap=30)
        drawBoundingBox(results, "main_image.jpg")
       
       
        #st.image(img, caption="Detection Results")
	
	
       

  
    
    

if __name__ == '__main__':
    main()
