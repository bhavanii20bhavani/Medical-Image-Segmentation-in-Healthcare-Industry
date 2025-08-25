import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from skfuzzy import control as ctrl
import cv2


def create_fuzzy_system():
    # Define fuzzy variables and their ranges
    quality = ctrl.Antecedent(np.arange(0, 256, 1), 'quality')
    brightness = ctrl.Antecedent(np.arange(0, 256, 1), 'brightness')
    segment = ctrl.Consequent(np.arange(0, 256, 1), 'segment')

    # Define fuzzy membership functions
    quality['low'] = fuzz.trimf(quality.universe, [0, 0, 128])
    quality['high'] = fuzz.trimf(quality.universe, [128, 255, 255])
    brightness['dark'] = fuzz.trimf(brightness.universe, [0, 0, 128])
    brightness['bright'] = fuzz.trimf(brightness.universe, [128, 255, 255])
    segment['poor'] = fuzz.trimf(segment.universe, [0, 0, 128])
    segment['good'] = fuzz.trimf(segment.universe, [128, 255, 255])

    # Define fuzzy rules
    rule1 = ctrl.Rule(quality['low'] & brightness['dark'], segment['poor'])
    rule2 = ctrl.Rule(quality['high'] & brightness['bright'], segment['good'])
    rule3 = ctrl.Rule(quality['low'] & brightness['bright'], segment['poor'])
    rule4 = ctrl.Rule(quality['high'] & brightness['dark'], segment['good'])

    # Create control system and simulation
    segmentation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    segmentation_sim = ctrl.ControlSystemSimulation(segmentation_ctrl)

    return segmentation_sim


def create_unet_model(input_shape, sol):
    inputs = layers.Input(shape=input_shape)
    Activation = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    c4 = layers.Conv2D(64, (3, 3), activation=Activation[int(sol[2])], padding='same')(u1)  # 'relu'

    u2 = layers.UpSampling2D((2, 2))(c4)
    c5 = layers.Conv2D(32, (3, 3), activation=Activation[int(sol[2])], padding='same')(u2)  # 'relu'

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c5)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def adaptive_fuzzy_ann_training(neural_network, fuzzy_system, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=32):
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            # Apply fuzzy system to preprocess input data
            preprocessed_X_batch = []
            for img in X_batch:
                img_uint8 = cv2.convertScaleAbs(img)  # Convert image to 8-bit unsigned integer format
                fuzzy_system.input['quality'] = np.mean(img_uint8)
                hsv_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
                fuzzy_system.input['brightness'] = np.mean(hsv_img[:, :, 2])

                # Debugging information
                print(f"Input quality: {np.mean(img_uint8)}, Input brightness: {np.mean(hsv_img[:, :, 2])}")

                fuzzy_system.compute()
                segment_quality = fuzzy_system.output['segment']
                preprocessed_X_batch.append(img * (segment_quality / 255.0))

            preprocessed_X_batch = np.array(preprocessed_X_batch)

            # Train neural network on preprocessed data
            neural_network.fit(preprocessed_X_batch, Y_batch, epochs=1, verbose=0)

        # Validate the model
        val_loss, val_accuracy = neural_network.evaluate(X_val, Y_val, verbose=0)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')


def segment_image(image, neural_network, fuzzy_system):
    # Convert image to 8-bit unsigned integer format
    img_uint8 = cv2.convertScaleAbs(image)

    # Apply fuzzy system to preprocess input data
    fuzzy_system.input['quality'] = np.mean(img_uint8)
    hsv_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
    fuzzy_system.input['brightness'] = np.mean(hsv_img[:, :, 2])

    # Debugging information
    print(f"Input quality: {np.mean(img_uint8)}, Input brightness: {np.mean(hsv_img[:, :, 2])}")

    fuzzy_system.compute()
    segment_quality = fuzzy_system.output['segment']
    preprocessed_image = img_uint8 * (segment_quality / 255.0)

    # Predict segmentation using neural network
    prediction = neural_network.predict(np.expand_dims(preprocessed_image, axis=0))
    return prediction


def Model_Ada_F_ANN(Images, Targets, sol=None):
    if sol is None:
        sol = [5, 5, 50]
    X_train, X_val, Y_train, Y_val = train_test_split(Images, Targets,
                                       random_state=104,
                                       test_size=0.25,
                                       shuffle=True)
    fuzzy_system = create_fuzzy_system()
    input_shape = (128, 128, 3)

    Train_Temp = np.zeros((X_train.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(X_train.shape[0]):
        Train_Temp[i, :] = np.resize(X_train[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((X_val.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(X_val.shape[0]):
        Test_Temp[i, :] = np.resize(X_val[i], (input_shape[0], input_shape[1], input_shape[2]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Train_Temp = np.zeros((Y_train.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Y_train.shape[0]):
        Train_Temp[i, :] = np.resize(Y_train[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_Y = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((Y_val.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Y_val.shape[0]):
        Test_Temp[i, :] = np.resize(Y_val[i], (input_shape[0], input_shape[1], input_shape[2]))
    Test_Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    unet_model = create_unet_model(input_shape, sol)
    # Train the adaptive fuzzy ANN
    adaptive_fuzzy_ann_training(unet_model, fuzzy_system, Train_X.astype(np.uint8), Train_Y, Test_X.astype(np.uint8),
                                Test_Y, epochs=int(sol[1]))
    segmented_images = []
    for image in X_val:
        segmented_image = segment_image(image.astype(np.uint8), unet_model, fuzzy_system)
        segmented_images.append(segmented_image)
    Predicted = [(img > 0.5).astype(np.uint8) * 255 for img in segmented_images]
    Predicted = np.asarray(Predicted)
    Pred = Predicted[:, 0, :, :, 0]
    return Pred
