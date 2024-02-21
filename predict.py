import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def predict_image_class(model_path, image_path):
    model = load_model(model_path)

    input_image = load_and_preprocess_image(image_path)

    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)

    print(f'Predicted class: {predicted_class}')

if __name__ == '__main__':
    model_path = 'path/to/your/trained/model.h5'
    input_image_path = 'path/to/your/input/image.jpg'
    
    predict_image_class(model_path, input_image_path)
