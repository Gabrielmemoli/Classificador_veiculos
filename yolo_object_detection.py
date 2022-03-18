import cv2
import numpy as np
import glob
import random
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny_final.weights", "yolov3-tiny.cfg")

# Name custom object
classes = ["bike", "carro", "moto", "pessoa", "cone"]

# Images path
images_path = glob.glob(r"C:\Users\USER\Desktop\imagens positivas\*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def ler_imagem(img):
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    #cv2.imshow("Image", img)
    return img
    key = cv2.waitKey(0)


# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
#for img_path in images_path:
    # Loading image
  #  imagem = cv2.imread(img_path)
   # ler_imagem(imagem)

while True:
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    imagem = ler_imagem(frame)
    cv2.imshow("Image", imagem)
    key = cv2.waitKey(10)
    #time.sleep(0.3)


cv2.destroyAllWindows()
