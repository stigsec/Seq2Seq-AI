import numpy as np
import tensorflow as tf
import pickle

with open('models\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    inputs = []
    outputs = []
    for i in range(0, len(lines), 2):
        inputs.append(lines[i].strip())
        outputs.append(lines[i+1].strip())
    return inputs, outputs

evaluation_dataset = "evaluation\\questions.txt"
evaluation_input_texts, evaluation_target_texts = load_data(evaluation_dataset)

max_seq_length = 10
def preprocess_data(inputs, outputs, tokenizer):
    input_seqs = tokenizer.texts_to_sequences(inputs)
    output_seqs = tokenizer.texts_to_sequences(outputs)
    input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, maxlen=max_seq_length, padding='post')
    output_seqs = tf.keras.preprocessing.sequence.pad_sequences(output_seqs, maxlen=max_seq_length, padding='post')
    return input_seqs, output_seqs

evaluation_encoder_input_data, evaluation_decoder_input_data = preprocess_data(evaluation_input_texts, evaluation_target_texts, tokenizer)
evaluation_decoder_target_data = np.roll(evaluation_decoder_input_data, -1, axis=1)
evaluation_decoder_target_data[:, -1] = tokenizer.word_index['<end>']

loaded_model = tf.keras.models.load_model('models\\seq2seq_model.h5')
evaluation_loss = loaded_model.evaluate([evaluation_encoder_input_data, evaluation_decoder_input_data], evaluation_decoder_target_data, verbose=1)
print("Evaluation Loss:", evaluation_loss)