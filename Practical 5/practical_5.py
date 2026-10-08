import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Lambda
import tensorflow.keras.backend as K
import numpy as np 

# --- a. Data Preparation --- 
# Sample corpus 
corpus = """ 
The quick brown fox jumps over the lazy dog. 
The dog barks, and the fox runs away. 
A quick brown rabbit also jumps. 
""" 

# Tokenize the corpus 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts([corpus]) 
word_index = tokenizer.word_index 
vocab_size = len(word_index) + 1 # +1 for padding/unknown words 
print(f"Vocabulary: {word_index}") 
print(f"Vocabulary size: {vocab_size}") 

# Convert corpus to sequences of integers 
sequences = tokenizer.texts_to_sequences([corpus])[0] 
print(f"Sequences: {sequences}") 

# --- b. Generate Training Data --- 
window_size = 2 
data_cbow = [] 
target_cbow = [] 

for i, word in enumerate(sequences): 
    context_start = max(0, i - window_size) 
    context_end = min(len(sequences), i + window_size + 1) 
    context_indices = sequences[context_start:i] + sequences[i+1:context_end] 
    
    if context_indices: 
        data_cbow.append(context_indices) 
        target_cbow.append(word) 

# Pad sequences so they all have the same length
max_context_length = window_size * 2
data_cbow = pad_sequences(data_cbow, maxlen=max_context_length, padding='post')
target_cbow = np.array(target_cbow) 

print(f"\nCBOW data shape: {data_cbow.shape}")
print(f"CBOW target shape: {target_cbow.shape}")

# --- c. Train Model --- 
embedding_dim = 10 

cbow_input = Input(shape=(max_context_length,), name='cbow_input')
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='word_embedding') 
embedded_contexts = embedding(cbow_input) 

def average_embeddings(embedded_sequences): 
    # Use K.mean to average the context embeddings
    return K.mean(embedded_sequences, axis=1) 

averaged_context = Lambda(average_embeddings, output_shape=(embedding_dim,), name='average_context')(embedded_contexts) 
cbow_output = Dense(vocab_size, activation='softmax', name='output_layer')(averaged_context) 

cbow_model = Model(inputs=cbow_input, outputs=cbow_output) 
cbow_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
cbow_model.summary() 

print("\nTraining the CBOW model...") 
history_cbow = cbow_model.fit(data_cbow, target_cbow, epochs=100, batch_size=16, verbose=0) 

# --- d. Output --- 
print("\nTraining finished. Extracting word embeddings.") 
word_embeddings = cbow_model.get_layer('word_embedding').get_weights()[0] 
print(f"Shape of word embeddings: {word_embeddings.shape}") 

print("\nWord Embeddings:") 
for word, i in word_index.items(): 
    print(f"'{word}': {word_embeddings[i][:5]}...") 
