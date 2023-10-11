from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# for tensorboard - training analysis
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# model
model= Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

## After training , we can save the weights of the trained model as: 

# model.save('action.h5')
## for loading the weights
# model.load_weights('action.h5')

## Evaluating the model with confusion matrix and accuracy as : 

# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat=np.argmax(yhat, axis=1).tolist()
# multilabel_confusion_matrix(ytrue, yhat)
# accuracy_score(ytrue, yhat)

