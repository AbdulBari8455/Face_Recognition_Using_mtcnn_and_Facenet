import numpy as np
import cv2
import mtcnn
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import Normalizer, LabelEncoder


# get encode
def face_preprocessor( frame, x1, y1, x2, y2, required_size=(160, 160)):
    """Method takes in frame, face coordinates and returns preprocessed image"""
    # 1. extract the face pixels
    face = frame[y1:y2, x1:x2]
    # 2. resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    # 3. scale pixel values
    face_pixels = face_array.astype('float32')
    # 4. standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # 5. transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # 6. get face embedding
    yhat = face_encoder.predict(samples)
    face_embedded = yhat[0]
    # 7. normalize input vectors
    in_encoder = Normalizer(norm='l2')
    X = in_encoder.transform(face_embedded.reshape(1, -1))
    return X


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    return (x1, y1), (x2, y2)
def face_classifier(X):
    """Methods takes in preprocessed images ,classifies and returns predicted Class label and probability"""
    # predict
    yhat = model.predict(X)
    label = yhat[0]
    print(label)
    yhat_prob = model.predict_proba(X)
    probability = round(yhat_prob[0][label], 2)
    trainy = data['arr_1']
    # predicted label decoder
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    predicted_class_label = out_encoder.inverse_transform(yhat)
    label = predicted_class_label[0]
    return label, str(probability)

def plt_show(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


encoder_model = 'Path to facenet_keras.h5'
data = np.load('Path to FaceFeature-embeddings.npz')
svm_model = pickle.load(open(" Path to model.sav", 'rb'))
face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)
test_res_path = 'Path of Resultant Image' 
test_img_path='Path Of image'     
recognition_t = 0.3
required_size = (160, 160)

face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)

img = cv2.imread(test_img_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detector.detect_faces(img_rgb)
for res in results:
    pt_1,pt_2 = get_face(img_rgb, res['box'])
    X = face_preprocessor(img_rgb , pt_1[0], pt_1[1], pt_2[0], pt_2[1], required_size=(160, 160))
    name = 'unknown'
    label, probability = face_classifier(X)

    if probability <= "0.95":
        cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
        cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    else:
        cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
        cv2.putText(img, label + "  " + probability, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 200, 200), 2)

cv2.imwrite(test_res_path, img)
plt_show(img)
