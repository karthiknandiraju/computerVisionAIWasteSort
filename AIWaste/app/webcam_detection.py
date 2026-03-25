import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/waste_classifier.h5")

classes = [
    'aerosol_cans',
    'aluminum_food_cans',
    'aluminum_soda_cans',
    'cardboard_boxes',
    'cardboard_packaging',
    'clothing',
    'coffee_grounds',
    'disposable_plastic_cutlery',
    'eggshells',
    'food_waste',
    'glass_beverage_bottles',
    'glass_cosmetic_containers',
    'glass_food_jars',
    'magazines',
    'newspaper',
    'office_paper',
    'paper_cups',
    'plastic_cup_lids',
    'plastic_detergent_bottles',
    'plastic_food_containers',
    'plastic_shopping_bags',
    'plastic_soda_bottles',
    'plastic_straws',
    'plastic_trash_bags',
    'plastic_water_bottles',
    'shoes',
    'steel_food_cans',
    'styrofoam_cups',
    'styrofoam_food_containers',
    'tea_bags'
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(frame,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)
    label = classes[np.argmax(pred)]

    cv2.putText(frame,label,(20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("AI Waste Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
