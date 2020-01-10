"""
Created by haiphung106
"""
import tensorflow as tf
import numpy as np
import io
from keras.layers import Dense, Dropout, Activation
from keras import metrics
from keras.layers import SimpleRNN
import os
import unicodedata
import time
import matplotlib.pyplot as plt
import sys

# tf.enable_eager_execution()

totalTime = time.time()


def i_to_c(arr):
    word = []
    for i in arr:
        word.append(int_to_cab[i])
    return (repr(''.join(word)))


data_URL = 'shakespeare_train.txt'
with io.open(data_URL, 'r', encoding='utf8') as f:
    text = f.read()

data_URL = 'shakespeare_valid.txt'
with io.open(data_URL, 'r', encoding='utf8') as f:
    textValidation = f.read()

"""
Reduce validation 
"""
vocab = set(text)
vocabB = set(textValidation)
vocab.union(vocabB)

vocab_to_int = {u: i for i, u in enumerate(vocab)}
int_to_cab = dict(enumerate(vocab))

validation_data = np.array([vocab_to_int[c] for c in textValidation], dtype=np.int32)
train_data = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text) // seq_length
validation_per_epoch = len(textValidation) // seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(train_data)
validation_char_dataset = tf.data.Dataset.from_tensor_slices(validation_data)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
validation_sequences = validation_char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)
validation = validation_sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE
step_per_Validationepoch = validation_per_epoch // BATCH_SIZE

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
validation = validation.batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 512

cellType = sys.argv[1]
print("")
if (cellType == "LSTM"):
    print("Using CuDNNLSTM")
    rnn = tf.keras.layers.LSTM
elif (cellType == "GRU"):
    print("Using CuDNNGRU")
    rnn = tf.keras.layers.GRU
else:
    print("Using SimpleRNN")
    rnn = tf.keras.layers.SimpleRNN

# define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[BATCH_SIZE, None]))
model.add(rnn(rnn_units, recurrent_initializer='glorot_uniform', stateful=True, return_sequences=True))
model.add(tf.keras.layers.Dense(vocab_size))

print(model.summary())


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    # return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)


def lossValue(labels, predicted):
    predictedLabel = np.argmax(predicted, axis=2)
    loss = 0

    for i in range(BATCH_SIZE):
        for s in range(seq_length):
            predicted[i, s] = softmax(predicted[i, s])
            loss += np.log(predicted[i, s, labels[i, s]])

    return -loss / (BATCH_SIZE * seq_length)


model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
EPOCHS = 30
# history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

print(steps_per_epoch)
print(step_per_Validationepoch)
print(cellType + " " + str(rnn_units) + " " + str(seq_length))
learningCurve = []
validationCurve = []
validationAccuracy = []
trainingAccuracy = []

for i in range(EPOCHS):
    lossVal = 0
    accVal = 0
    lossVal2 = 0
    start_time = time.time()
    for input_example, target_example in dataset:
        currentLoss = model.train_on_batch(input_example, target_example)

        yhat = model(input_example)
        y_pred_cls = np.argmax(yhat, axis=2)
        correct_prediction = np.sum(y_pred_cls == target_example.numpy())
        currentAcc = correct_prediction / (BATCH_SIZE * seq_length)

        lossVal2 += lossValue(target_example.numpy(), yhat.numpy())
        lossVal += currentLoss
        accVal += currentAcc

    print("time / epoch : " + str(int(time.time() - start_time)) + " seconds")
    print("Loss Epoch " + str(i) + " :", lossVal / (steps_per_epoch))
    print("Loss2 Epoch " + str(i) + " :", lossVal2 / (steps_per_epoch))
    print("Acc Epoch " + str(i) + " :", np.round(accVal * 100.0 / (steps_per_epoch), 2), " %")

    learningCurve.append(lossVal / (steps_per_epoch))
    trainingAccuracy.append(np.round(accVal * 100.0 / (steps_per_epoch), 2))

    for input_example, target_example in dataset.take(1):
        yhat = model(input_example)
        y_pred_cls = np.argmax(yhat, axis=2)
        print("Input:  ", i_to_c(input_example[0].numpy()))
        print("Output: ", i_to_c(target_example[0].numpy()))
        print("Predict:", i_to_c(y_pred_cls[0]))

    # validation
    lossVal = 0
    accVal = 0
    for input_example, target_example in validation:
        yhat = model(input_example)
        y_pred_cls = np.argmax(yhat, axis=2)
        correct_prediction = np.sum(y_pred_cls == target_example.numpy())
        currentAcc = correct_prediction / (BATCH_SIZE * seq_length)
        currentLoss = loss(target_example, yhat)  # lossValue(target_example.numpy(), y_pred_cls,yhat.numpy())
        lossVal += np.sum(currentLoss) / (BATCH_SIZE * seq_length)
        accVal += correct_prediction / (BATCH_SIZE * seq_length)

    print("Validation Loss " + str(i) + " :", lossVal / (step_per_Validationepoch))
    print("Validation Accu " + str(i) + " :", np.round(accVal * 100.0 / (step_per_Validationepoch), 2), " %")

    validationCurve.append(lossVal / (step_per_Validationepoch))
    validationAccuracy.append(np.round(accVal * 100.0 / (step_per_Validationepoch), 2))

totalTime = time.time() - totalTime

print("Total Training TimeL: " + str(totalTime / 60) + " minutes")

# visualize
plt.plot(learningCurve, color='g', label='Training')
plt.plot(validationCurve, color='b', label='Validation')
plt.legend(loc='best')
plt.title("Learning Curve with " + cellType)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("LearningCurve" + cellType + str(rnn_units) + "_seq" + str(seq_length) + ".png", bbox_inches="tight",
            dpi=150)
plt.show()
plt.clf()

# visualize
plt.plot(trainingAccuracy, color='g', label='Training')
plt.plot(validationAccuracy, color='b', label='Validation')
plt.legend(loc='best')
plt.title("Accuracy with " + cellType)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("AccuracyCurve" + cellType + str(rnn_units) + "_seq" + str(seq_length) + ".png", bbox_inches="tight",
            dpi=150)
plt.show()
plt.clf()

print(learningCurve)
print(validationCurve)
print(trainingAccuracy)
print(validationAccuracy)

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]))
model2.add(rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True))
model2.add(tf.keras.layers.Dense(vocab_size))
model2.set_weights(model.get_weights())

model2.build(tf.TensorShape([1, None]))

model2.summary()


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # You can change the start string to experiment
    # start_string =""

    # Converting our start string to numbers (vectorizing)
    input_eval = [vocab_to_int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(int_to_cab[predicted_id])

    return (start_string + ''.join(text_generated))


generatedText = generate_text(model2, start_string="ROMEO:")
print(generatedText)
f = open("ROMEOGenerated" + cellType + str(rnn_units) + "_seq" + str(seq_length) + ".txt", 'w')
f.write(generatedText)
f.close()

generatedText = generate_text(model2, start_string="JULIET:")
print(generatedText)
f = open("JULIETGenerated" + cellType + str(rnn_units) + "_seq" + str(seq_length) + ".txt", 'w')
f.write(generatedText)
f.close()
