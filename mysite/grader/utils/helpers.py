import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import math
from collections import Counter

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model.wv[word])        
    featureVec = np.divide(featureVec,num_words)
    return featureVec
    # featureVec = np.zeros((num_features,), dtype="float32")
    # nwords = 0
    
    # for word in words:
    #     if word in model:
    #         nwords += 1
    #         featureVec = np.add(featureVec, model[word])
            
    # featureVec = np.divide(featureVec,nwords)
    # return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

def tokenize(essay):
  """Tokenize essay into words, remove stopwords and lowercasing"""
  essay = re.sub(r'[^\w\s]', '', essay)  
  essay = essay.lower()  
  essay = essay.split()
  essay = [word for word in essay if word not in stopwords.words('english')]
  return essay
    
def compute_words_rate(corpus):
  """Compute generic rates G_i from the entire corpus."""
  word_counts = Counter()
  total_words = 0
  for essay in corpus:
    word_counts.update(essay)
    total_words += len(essay)
  return {word: count / total_words for word, count in word_counts.items()}

from sklearn.metrics.pairwise import cosine_similarity

def calculate_z_scores(cosine_similarities, mean, std_dev):
    """Calculate the z-scores based on cosine similarities."""
    return (cosine_similarities - mean) / std_dev 

def modelA(novel_essays, novel_essays_prompt, X_train, mean_max_cos_sim, std_dev_max_cos_sim, mean_prompt_cos, std_dev_prompt_cos):
  '''
  predict off-topicity using cosine similarity and z-scores
  '''
  # cos similarity between novel essay and training essays
  max_cos_sim = np.max(cosine_similarity(novel_essays, X_train))

  # cos similarity between novel essay and its corresponding prompt
  prompt_cos = cosine_similarity(novel_essays, novel_essays_prompt)

  z_scores_max_cos_sim = calculate_z_scores(max_cos_sim, mean_max_cos_sim, std_dev_max_cos_sim)
  z_scores_prompt_cos = calculate_z_scores(prompt_cos, mean_prompt_cos, std_dev_prompt_cos)

  return z_scores_max_cos_sim, z_scores_prompt_cos

def compute_psi(essay, generic_rate, prompt_specific_rate):
    """Compute the Prompt-Specific Index for a single essay."""
    N = len(essay)  
    if N == 0:  # Avoid division by zero
        return 0

    psi_sum = 0
    for word in essay:
        G_i = generic_rate.get(word, 0)
        S_i = prompt_specific_rate.get(word, 0)
        psi_sum += np.sqrt(S_i * (1 - G_i))

    return psi_sum / N
