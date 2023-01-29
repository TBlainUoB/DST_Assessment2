import gensim
import pandas as pd
from nltk.tokenize import word_tokenize
import string
import nltk
import numpy as np


# nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    text = ' '.join(text.split())
    return text


# Load the dataset with movie titles and plots
df = pd.read_csv('PreProcessedData.csv')

# Preprocess and tokenize the text
df['plot_tokens'] = df['Plot'].apply(lambda x: word_tokenize(preprocess_text(x)))
df['title_tokens'] = df['Plot'].apply(lambda x: word_tokenize(preprocess_text(x)))

# Train the word2vec model
model = gensim.models.Word2Vec(df['plot_tokens'], min_count=1)
model_title = gensim.models.Word2Vec(df['title_tokens'], min_count=1)

# Access the word vectors
vector = model.wv['sentence']
vector_title = model_title.wv['sentence']

# Perform similarity operations
similar = model.wv.most_similar('sentence')
similar_title = model_title.wv.most_similar('sentence')

model.save("word2vec.model")
np.save("word2vec_vectors.npy", model.wv.vectors)

average_word_vectors = []
average_word_vectors_title = []
for plot in df['plot_tokens']:
    average_word_vectors.append(np.mean(model.wv[plot], axis=0))
for title in df['title_tokens']:
    average_word_vectors_title.append(np.mean(model_title.wv[title], axis=0))

# Convert the arrays of word vectors into dataframes
average_word_vectors_df = pd.DataFrame(average_word_vectors)
average_word_vectors_title_df = pd.DataFrame(average_word_vectors_title)

# Give the columns of the dataframes meaningful names
average_word_vectors_df.columns = [f'avg_wordvec_{i}' for i in range(average_word_vectors_df.shape[1])]
average_word_vectors_title_df.columns = [f'avg_title_wordvec_{i}' for i in range(average_word_vectors_title_df.shape[1])]

# Concatenate the dataframes with the original dataframe
df = pd.concat([df, average_word_vectors_df, average_word_vectors_title_df], axis=1)


'''
df['average_word_vectors'] = average_word_vectors
df['average_title_word_vectors'] = average_word_vectors_title
'''
df.drop('Plot', axis=1, inplace=True)
df.drop('plot_tokens', axis=1, inplace=True)
df.drop('Title', axis=1, inplace=True)
df.drop('title_tokens', axis=1, inplace=True)

df.to_csv("Model_W2V.csv", index=False)
