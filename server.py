from flask import Flask, request, jsonify
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.contours import sort_contours
from io import BytesIO


def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]
    return conts

def extract_roi(img, x, y, w, h, margin=2):
    roi = img[y - margin:y + h + margin, x - margin:x + w + margin]
    return roi

def thresholding(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

def resize_img(img, w, h):
    if w > h:
        resized = imutils.resize(img, width=28)
    else:
        resized = imutils.resize(img, height=28)

    (h, w) = resized.shape
    dX = int(max(0, 28 - w) / 2.0)
    dY = int(max(0, 28 - h) / 2.0)

    filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    filled = cv2.resize(filled, (28, 28))
    return filled

def normalization(img):
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def process_box(gray, x, y, w, h):
    roi = extract_roi(gray, x, y, w, h)
    thresh = thresholding(roi)
    (h, w) = thresh.shape
    resized = resize_img(thresh, w, h)
    normalized = normalization(resized)
    characters.append((normalized, (x, y, w, h)))

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img = cv2.resize(thresh, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.reshape(img, (1, 28, 28, 1))
    return img

characters = []
boxes = []
def group_characters_into_words(boxes):
    if not boxes:
        return []

    # Calculate the average width of characters and spaces between them
    widths = [w for _, _, w, _ in boxes]
    spaces = [boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes) - 1)]
    
    if not spaces:
        return [boxes]  # If there's only one box, it's one word

    avg_space = np.mean(spaces)
    threshold = avg_space * 1.5  # Adjust the multiplier as necessary

    words = []
    current_word = [boxes[0]]
    last_x = boxes[0][0] + boxes[0][2]

    for i in range(1, len(boxes)):
        x, y, w, h = boxes[i]
        if (x - last_x) > threshold:
            words.append(current_word)
            current_word = []
        current_word.append((x, y, w, h))
        last_x = x + w

    if current_word:
        words.append(current_word)

    return words

def predictImage(img):
    global characters, boxes
    characters = []
    boxes = []
    # Initialize lists
    model = load_model(r"C:\Users\Enes\Desktop\OCR\network")

    # Character list
    character_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    character_list = [l for l in character_list]
    # Initialize lists

    img = np.array(img)
    print("image type is",type(img))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    invertion = 255 - adaptive
    dilation = cv2.dilate(invertion, np.ones((3,3)))

    # Find contours
    conts = find_contours(dilation.copy())
    min_w, max_w = 4, 160
    min_h, max_h = 14, 140
    
    # Process each contour
    for c in conts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
            process_box(gray, x, y, w, h)

    # Extract boxes and pixels
    boxes = [box[1] for box in characters]
    pixels = np.array([pixel[0] for pixel in characters], dtype='float32')
    
    # Make predictions
    predictions = model.predict(pixels)

    # Group characters into words
    words = group_characters_into_words(boxes)

    results = []
    for word in words:
        word_text = ""
        for (x, y, w, h) in word:
            idx = boxes.index((x, y, w, h))
            prediction = predictions[idx]
            i = np.argmax(prediction)
            probability = prediction[i]
            character = character_list[i]
            word_text += character
        results.append(word_text)
    return results


app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
        results = predictImage(img)
    
        return jsonify({"processed_image": results})
    except Exception as e:
        print("upload error",str(e))
        return jsonify({"error": str(e)}), 500
    
def runServer():
    app.run(host='0.0.0.0', port=8000)

@app.route('/home', methods=['GET'])
def run():
    return jsonify({"error":"2sdgasg5"}), 200

if __name__ == '__main__':
    runServer()


