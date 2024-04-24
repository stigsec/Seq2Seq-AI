import numpy as np
import tensorflow as tf
import pickle

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

print('Loading model')
with tf.keras.utils.custom_object_scope({'ExpandDimsLayer': ExpandDimsLayer}):
    model = tf.keras.models.load_model('models\\seq2seq_model.h5')
print('Loading model completed')

print('Loading tokenizer')
with open('models\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print('Loading tokenizer completed')

max_seq_length = 10

def preprocess_input(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=max_seq_length, padding='post')
    return padded_sequence

def generate_response(input_text):
    input_seq = preprocess_input(input_text)
    output_seq = np.zeros((1, max_seq_length))
    output_seq[0, 0] = tokenizer.word_index['<start>']
    
    for i in range(1, max_seq_length):
        predictions = model.predict([input_seq, output_seq])
        predicted_id = np.argmax(predictions[0, i-1, :])
        output_seq[0, i] = predicted_id
        if predicted_id == tokenizer.word_index['<end>']:
            break
    
    print("Output sequence:", output_seq)
    
    output_text = []
    for token_id in output_seq[0]:
        if token_id == 0:
            break
        word = tokenizer.index_word.get(token_id, '')
        if word == '<end>':
            break
        output_text.append(word)
    
    print("Output text:", output_text)
    
    return ' '.join(output_text)

print("Ready")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("exitting..")
        break
    response = generate_response(user_input)
    print("Bot:", response)
