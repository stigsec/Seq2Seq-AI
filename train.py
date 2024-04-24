import numpy as np
import tensorflow as tf
import pickle

num_epochs = 250
batch_size = 128
learning_rate = 0.001
max_seq_length = 10
embedding_size = 256
hidden_size = 512
vocab_size = None
dataset = "dataset\\twitter\\ready_twittercorpus.txt"

name = "settings.txt"
with open(name, 'w') as file:
    file.write(f"num_epochs = {num_epochs}\n")
    file.write(f"batch_size = {batch_size}\n")
    file.write(f"learning_rate = {learning_rate}\n")
    file.write(f"max_seq_length = {max_seq_length}\n")
    file.write(f"embedding_size = {embedding_size}\n")
    file.write(f"hidden_size = {hidden_size}\n")
    file.write(f"dataset = {dataset}\n")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    inputs = []
    outputs = []
    for i in range(0, len(lines), 2):
        inputs.append(lines[i].strip())
        outputs.append(lines[i+1].strip())
    return inputs, outputs

def preprocess_data(inputs, outputs, tokenizer):
    input_seqs = tokenizer.texts_to_sequences(inputs)
    output_seqs = tokenizer.texts_to_sequences(outputs)
    input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, maxlen=max_seq_length, padding='post')
    output_seqs = tf.keras.preprocessing.sequence.pad_sequences(output_seqs, maxlen=max_seq_length, padding='post')
    return input_seqs, output_seqs

def build_encoder(input_shape, vocab_size):
    encoder_inputs = tf.keras.layers.Input(shape=input_shape)
    encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(hidden_size, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    return encoder_inputs, encoder_states

def build_decoder(target_shape, vocab_size, encoder_states):
    decoder_inputs = tf.keras.layers.Input(shape=target_shape)
    decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)(decoder_inputs)
    
    decoder_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    return decoder_inputs, decoder_outputs

def build_model(input_shape, target_shape, vocab_size):
    encoder_inputs, encoder_states = build_encoder(input_shape, vocab_size)
    decoder_inputs, decoder_outputs = build_decoder(target_shape, vocab_size, encoder_states)
    
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

print('Loading data')
input_texts, target_texts = load_data(dataset)
print('Loading data completed')

print('Tokenizing data')
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)

if '<start>' not in tokenizer.word_index:
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1

if '<end>' not in tokenizer.word_index:
    tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

vocab_size = len(tokenizer.word_index) + 1
print('Tokenizing data completed')

print('Saving tokenizer data')
with open('models\\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saving tokenizer data completed')

print('Preprocessing data')
encoder_input_data, decoder_input_data = preprocess_data(input_texts, target_texts, tokenizer)
decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
decoder_target_data[:, -1] = tokenizer.word_index['<end>']
print('Preprocessing data completed')

print('Building and compiling the model')
model = build_model((max_seq_length,), (max_seq_length,), vocab_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy')
print('Building and compiling the model completed')

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}/{num_epochs}')
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1)
    
    print('Epoch Loss:', history.history['loss'])
    
    print(f'Epoch {epoch+1}/{num_epochs} completed')
    
print('Saving the model')
model.save('models\\seq2seq_model.h5')
print('Saving the model completed')