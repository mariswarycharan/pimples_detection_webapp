import shutil
import cv2
import torch
import streamlit as st

st.header("Detect the pimples")
button_pim = st.button("Click here")
image = st.image([])
exp_change_num = 0


if button_pim:
    stop = st.button("STOP")
    video = cv2.VideoCapture(0)
    model = torch.hub.load(r'yolov5', 'custom',source="local", path = r"pimples_weights.pt" )
    classes_list = model.names
    detected_object_name_list = []

    while True:
        
        ret,frame = video.read()
        frame = cv2.flip(frame,1)
        results = model(frame, size=640)
        results.save()
        get_array = results.xyxy[0]
        get_array = get_array.tolist()
        
        if len(get_array) == 0:
            pass
        else:
            for i in get_array:
                last = classes_list[round(i[-1])]
                detected_object_name_list.append(last)
        
        if len(detected_object_name_list) == 0:
            print("No odject is detected!!!")
        else:
            print("Detected objects are :","|".join(detected_object_name_list))
        
        detected_object_name_list = []
        
        output_img = cv2.imread(r"runs\detect\exp\image0.jpg")
        image.image(output_img)
        shutil.rmtree(r"runs")