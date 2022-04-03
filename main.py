import datetime
import pytesseract
from pytesseract import Output
import cv2
import os


def crop_passport_photo(img):
    ori_img = img.copy()
    img_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected_faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=6)

    for (x, y, width, height) in detected_faces:
        x, y, width, height = x-50, y-50, width+100, height+100
        x1, y1, x2, y2 = x, y, x+width, y+height
        crop_img = ori_img[y1:y2, x1:x2]
        current_timestamp = datetime.datetime.now()
        file_name = f'Results/{str(current_timestamp)}.png'
        cv2.imwrite(file_name, crop_img)


def get_words_structure(img):
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(data['level'])
    words_structure = []
    for i in range(n_boxes):
        coords = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        text = (data["text"][i])
        confidence = (data["conf"][i])
        if text != "":
            if text != " ":
                words_structure.append({"coords": coords, "text": text,
                                        "confidence": confidence})
    return words_structure


def ocr_driving_licence(img):
    img = img[0:1120, 100:1900]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img, (5, 5), 0)
    img1 = cv2.adaptiveThreshold(img1, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    words_structure = get_words_structure(img)
    words_structure1 = get_words_structure(img1)
    words_structure.extend(words_structure1)

    words_structure = [dict(t) for t in {tuple(d.items()) for d in words_structure}]

    for words in words_structure:
        with open('Results/Results.txt', 'a') as results_file:
            results_file.write(f''' \n \n {words}
                ''')
            x, y, w, h = words["coords"]
            text = words["text"]
            # if words["confidence"] >= 50:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite('Results/Result.png', img)

try:
    if not os.path.exists('Results'):
        os.makedirs('Results')

    image_file = '1648136552345-1.JPEG'
    img = cv2.imread(image_file)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    crop_passport_photo(img)
    ocr_driving_licence(img)
except Exception as error:
    print(error)



