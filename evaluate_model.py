import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def evaluate_model(model_path, test_data_generator):
    model = load_model(model_path)

    test_data, test_labels = test_data_generator
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    model_path = 'path/to/your/trained/model.h5'
    test_data_generator = get_test_data_generator()
    
    evaluate_model(model_path, test_data_generator)
