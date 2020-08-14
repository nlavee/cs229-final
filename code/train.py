import numpy as np


def batch_dot(Z1, Z2):
    """
    Computes the dot product between embeddings Z1 and Z2, as a batch.

    Args:
      Z1: A numpy array of shape (batch_size, embedding_size)
      Z2: A numpy array of shape (batch_size, embedding_size)

    Returns:
      A numpy array of shape (batch_size, ) containing the dot products
    """
    # *** START CODE HERE ***
    print("Entering Batch Dot")
    print(f"Z1: {Z1.shape}")
    print(f"Z2: {Z2.shape}")

    n, d = Z1.shape

    res = []

    for i in range(n):
        print(f"Z1[{i}]: {Z1[i]}")
        print(f"Z2[{i}]: {Z2[i]}")
        print(f"Dot product: {np.dot(Z1[i], Z2[i])}")
        res.append(np.dot(Z1[i], Z2[i]))

    res = np.array(res).reshape(n,)
    print(f'Batch_Dot: {res} \n({res.shape})')
    return res
    # *** END CODE HERE ***


def distance_loss(y_true, d):
    """
    Computes the average loss for the given batch's predictions d and respective
    true labels y_true (0 for similar-sounding and 1 for different-sounding).

    Args:
      y_true: A numpy array of shape (batch_size, ) containing true labels
      d: A numpy array of shape (batch_size, ) containing predicted distances

    Returns:
      The average loss for the given batch (a scalar)
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def accuracy(y_true, d):
    """
    Computes the average accuracy for the given batch's predictions d and respective
    true labels y_true (0 for similar-sounding and 1 for different-sounding).

    Args:
      y_true: A numpy array of shape (batch_size, ) containing true labels
      d: A numpy array of shape (batch_size, ) containing predicted distances

    Returns:
      The average accuracy for the given batch (a scalar)
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


########################################################################################################################

import random
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from sklearn.manifold import TSNE
from tqdm.auto import trange


def train(X1_train, X2_train, y_train, X1_dev, X2_dev, y_dev,
          num_hidden=25, num_epochs=8, batch_size=16, learning_rate=0.1):
    W = np.random.randn(X1_train.shape[1], num_hidden)
    history = defaultdict(list)
    with trange(num_epochs, desc='Epochs') as epoch_progress:
        for epoch in epoch_progress:
            gradient_descent_epoch(X1_train, X2_train, y_train, W, batch_size, learning_rate)
            train_state = forward(X1_train, X2_train, y_train, W)
            history['train_loss'].append(train_state['loss'])
            history['train_accuracy'].append(accuracy(y_train, train_state['d']))
            dev_state = forward(X1_dev, X2_dev, y_dev, W)
            history['dev_loss'].append(dev_state['loss'])
            history['dev_accuracy'].append(accuracy(y_dev, dev_state['d']))
            epoch_progress.set_postfix({k: v[-1] for k, v in history.items()})
    return W, history


def gradient_descent_epoch(X1, X2, y, W, batch_size, learning_rate):
    batch_start = 0
    for batch_end in range(batch_size, len(X1), batch_size):
        X1_batch = X1[batch_start:batch_end]
        X2_batch = X2[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]
        state = forward(X1_batch, X2_batch, y_batch, W)
        dW = clip_gradient(backward(state), max_norm=10)
        W -= learning_rate * dW
        batch_start = batch_end


def forward(X1, X2, y, W):
    Z1 = X1 @ W
    Z2 = X2 @ W
    d = 1.0 - batch_dot(Z1, Z2)
    loss = distance_loss(y, d)
    return {'X1': X1, 'X2': X2, 'Z1': Z1, 'Z2': Z2, 'd': d, 'loss': loss}


def backward(state):
    batch_size = len(state['X1'])
    dd = 2 * state['d'] - 1
    dZ1 = -state['Z2'].T @ dd
    dZ2 = -state['Z1'].T @ dd
    dW1 = state['X1'].T @ np.broadcast_to(dZ1, (batch_size, dZ1.shape[0]))
    dW2 = state['X2'].T @ np.broadcast_to(dZ2, (batch_size, dZ2.shape[0]))
    return (dW1 + dW2) / (2.0 * batch_size)


def clip_gradient(dW, max_norm):
    norm = np.linalg.norm(dW)
    return (np.clip(norm, 0, max_norm) / (np.finfo(float).eps + norm)) * dW


def plot_learning_curves(history):
    figure, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True, figsize=(10, 5))
    ax0.plot(history['train_loss'])
    ax0.plot(history['dev_loss'])
    ax0.set_ylabel('Loss')
    ax0.set_xlabel(f'Ran at {datetime.utcnow():%Y-%m-%d %H:%M:%S}')
    ax1.plot(history['train_accuracy'])
    ax1.plot(history['dev_accuracy'])
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    plt.legend(['Train', 'Dev'])
    plt.show()
    return figure


def plot_embeddings(words, W):
    sample_words = {w: represent_word(w) for w in words[100:150] + words[-100:]}
    embeddings = np.array(list(sample_words.values())) @ W
    return plot_tsne(list(sample_words.keys()), embeddings)


def represent_word(word):
    x = [0] * 3 * 26
    for i, c in enumerate(word):
        x[i * 26 + ord(c) - ord('a')] = 1
    return x


def plot_tsne(words, embeddings):
    model = TSNE(random_state=42)
    tsne_points = model.fit_transform(embeddings)
    x, y = zip(*tsne_points)
    figure = plt.figure(figsize=(15, 10))
    plt.scatter(x, y, alpha=0.5)
    for i, j, word in zip(x, y, words):
        plt.annotate(word, xy=(i, j), xytext=(3, 3), textcoords='offset points')
    plt.show()
    return figure


def print_phonetic_analogies(words, W):
    embeddings = np.array([represent_word(w) for w in words]) @ W
    embeddings_tree = KDTree(embeddings)

    def complete_analogy(x, y, z):
        embedding = embeddings[words.index(y)] - embeddings[words.index(x)] + embeddings[words.index(z)]
        _, i = embeddings_tree.query(embedding)
        return words[i]

    triads = [
        ('cap', 'coo', 'zap'),
        ('how', 'wow', 'hit'),
        ('yam', 'ham', 'yen'),
    ]
    print('Phonetic Analogies:')
    for triad in triads:
        forth = complete_analogy(*triad)
        print(f'"{triad[0]}" is to "{triad[1]}" as "{triad[2]}" is to "{forth}"')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    vars().update(
        {k: np.load(f'data/{k}.npy') for k in ('X1_train', 'X2_train', 'y_train', 'X1_dev', 'X2_dev', 'y_dev')})
    W, history = train(X1_train, X2_train, y_train, X1_dev, X2_dev, y_dev)
    plot_learning_curves(history).savefig('./learning-curves.pdf')
    with open('data/words') as in_file:
        words = [line.strip() for line in in_file]
    plot_embeddings(words, W).savefig('./embeddings.pdf')
    print('Plots were automatically saved to ./learning-curves.pdf and ./embeddings.pdf')
    print_phonetic_analogies(words, W)
