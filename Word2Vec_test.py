import gensim

# model = gensim.models.Word2Vec.load_word2vec_format("train_pure_small.vector",binary=False)

model = gensim.models.KeyedVectors.load_word2vec_format("train_pure_small.vector", binary=False)

model.most_similar("驾驶")

