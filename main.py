import os
from pathlib import Path
import pickle
import librosa
import librosa.feature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

labels = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

base_path = './data/genres_original'

file_path = Path('/path/to/your/file.txt')


def extract_librosa_data(out_path: Path):
    extracted_data = []
    for i, label in enumerate(labels):
        for filename in os.scandir(f'{base_path}/{label}'):
            if filename.is_file():
                if filename.name in ['.DS_Store', 'jazz.00054.wav']:
                    continue
                x, sr = librosa.load(f'{base_path}/{label}/{filename.name}')
                extracted_data.append((i, x, sr, filename.name))
    with open(out_path, 'wb') as f:
        pickle.dump(extracted_data, f)

    return extracted_data


def load_data(in_path: Path):
    with open(in_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def get_mfcc(y, sr):
    return np.array(librosa.feature.mfcc(y=y, sr=sr))


def get_mel_spectrogram(y, sr):
    return np.array(librosa.feature.melspectrogram(y=y, sr=sr))


def get_chroma_stft(y, sr):
    return np.array(librosa.feature.chroma_stft(y=y, sr=sr))


def get_chroma_cens(y, sr):
    return np.array(librosa.feature.chroma_cens(y=y, sr=sr))


def get_tonnetz(y, sr):
    return np.array(librosa.feature.tonnetz(y=y, sr=sr))


def get_zero_crossing_rate(y):
    return librosa.feature.zero_crossing_rate(y)


def get_spectral_bandwidth(y, sr):
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)


def get_spectral_centroid(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr)


def get_spectral_rolloff(y, sr):
    return librosa.feature.spectral_rolloff(y=y, sr=sr)


def get_spectral_contrast(y, sr):
    return librosa.feature.spectral_contrast(y=y, sr=sr)


def get_poly_features(y, sr):
    return librosa.feature.poly_features(y=y, sr=sr)


def get_tempo(y, sr):
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    prior_lognorm = stats.lognorm(loc=np.log(120), scale=120, s=1)
    return librosa.feature.tempo(
        onset_envelope=onset_envelope,
        sr=sr,
        aggregate=None,
        prior=prior_lognorm,
    )


def get_feature_stats(values):
    return {
        'mean': np.mean(values, axis=1),
        'std': np.std(values, axis=1),
        'skew': stats.skew(values, axis=1),
        'kurtosis': stats.kurtosis(values, axis=1),
        'median': np.median(values, axis=1),
        'min': np.min(values, axis=1),
        'max': np.max(values, axis=1),
    }


def concatenate_feature_stats(feature_stats):
    return np.concatenate((
        feature_stats['mean'],
        feature_stats['std'],
        feature_stats['skew'],
        feature_stats['kurtosis'],
        feature_stats['median'],
        feature_stats['min'],
        feature_stats['max']
    ))


def get_feature(y, sr):
    mfcc = get_mfcc(y, sr)
    feature_stats = get_feature_stats(mfcc)
    mfcc_feature = concatenate_feature_stats(feature_stats)

    mel_spectrogram = get_mel_spectrogram(y, sr)
    feature_stats = get_feature_stats(mel_spectrogram)
    mel_spectrogram_feature = concatenate_feature_stats(feature_stats)

    chroma_stft = get_chroma_stft(y, sr)
    feature_stats = get_feature_stats(chroma_stft)
    chroma_stft_feature = concatenate_feature_stats(feature_stats)

    chroma_cens = get_chroma_cens(y, sr)
    feature_stats = get_feature_stats(chroma_cens)
    chroma_cens_feature = concatenate_feature_stats(feature_stats)

    tonnetz = get_tonnetz(y, sr)
    feature_stats = get_feature_stats(tonnetz)
    tonnetz_feature = concatenate_feature_stats(feature_stats)

    zero_crossing_rate = get_zero_crossing_rate(y)
    feature_stats = get_feature_stats(zero_crossing_rate)
    zero_crossing_rate_feature = concatenate_feature_stats(feature_stats)

    spectral_bandwidth = get_spectral_bandwidth(y, sr)
    feature_stats = get_feature_stats(spectral_bandwidth)
    spectral_bandwidth_feature = concatenate_feature_stats(feature_stats)

    spectral_rolloff = get_spectral_rolloff(y, sr)
    feature_stats = get_feature_stats(spectral_rolloff)
    spectral_rolloff_feature = concatenate_feature_stats(feature_stats)

    spectral_centroid = get_spectral_centroid(y, sr)
    feature_stats = get_feature_stats(spectral_centroid)
    spectral_centroid_feature = concatenate_feature_stats(feature_stats)

    spectral_contrast = get_spectral_contrast(y, sr)
    feature_stats = get_feature_stats(spectral_contrast)
    spectral_contrast_feature = concatenate_feature_stats(feature_stats)

    poly_features = get_poly_features(y, sr)
    feature_stats = get_feature_stats(poly_features)
    poly_features_feature = concatenate_feature_stats(feature_stats)

    # Note: Tempo takes ages to populate, and doesn't improve the results much, if at all
    # tempo = get_tempo(y, sr)
    # tempo = np.expand_dims(tempo, axis=0)
    # feature_stats = get_feature_stats(tempo)
    # tempo_feature = concatenate_feature_stats(feature_stats)

    return np.concatenate((
        chroma_stft_feature,
        chroma_cens_feature,
        mel_spectrogram_feature,
        mfcc_feature,
        tonnetz_feature,
        zero_crossing_rate_feature,
        spectral_bandwidth_feature,
        spectral_centroid_feature,
        spectral_rolloff_feature,
        spectral_contrast_feature,
        poly_features_feature,
        # tempo_feature,
    ))


def generate_features(data, out_path: Path):
    xs = []
    ys = []
    filenames = []
    for label, y, sr, filename in data:
        features = get_feature(y, sr)
        xs.append(features)
        ys.append(label)
        filenames.append(filename)
    features = np.array(xs)
    labels = np.array(ys)

    feature_data = {"features": features, "labels": labels, "filenames": filenames}
    with open(out_path, 'wb') as f:
        pickle.dump(feature_data, f)

    return feature_data


def make_training_data(feature_data, out_path: Path):
    permutations = np.random.permutation(999)
    features = np.array(feature_data["features"])[permutations]
    labels = np.array(feature_data["labels"])[permutations]
    print(features)
    print(labels)
    features_train = features[0:900]
    labels_train = labels[0:900]
    features_test = features[900:999]
    labels_test = labels[900:999]
    data_set = (
        (labels_train, features_train),
        (labels_test, features_test)
    )
    with open(out_path, 'wb') as f:
        pickle.dump(data_set, f)
    return data_set


def build_model():
    model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
    model.compile(metrics=["accuracy"])

    return model


def train(model, data_set):
    return model.fit(
        x=data_set[0][1].tolist(),
        y=data_set[0][0].tolist(),
    )


def evaluate_model(model, data_set):
    return model.evaluate(x=data_set[1][1].tolist(), y=data_set[1][0].tolist(), return_dict=True)


if __name__ == '__main__':
    stored_data_path = Path('pickles/librosa_data.pkl')
    data = load_data(stored_data_path) \
        if stored_data_path.exists() \
        else extract_librosa_data(stored_data_path)

    # print(data)

    stored_feature_data_path = Path('pickles/features.pkl')
    feature_data = load_data(stored_feature_data_path) \
        if stored_feature_data_path.exists() \
        else generate_features(data, stored_feature_data_path)

    # print(feature_data)

    stored_training_data_path = Path('pickles/training_data.pkl')
    training_data = load_data(stored_training_data_path) \
        if stored_training_data_path.exists() \
        else make_training_data(feature_data, stored_training_data_path)

    # print(training_data)

    model = build_model()

    history = train(model, training_data)
    evaluation = evaluate_model(model, training_data)
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    logs = model.make_inspector().training_logs()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    plt.show()

    model.save('saved_models/random_forest_1')

    embeddings = model.predict(tf.constant(feature_data["features"]))

    print(embeddings)

    """
    Dimensionality Reduction with Neighborhood Components Analysis
    https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html#sphx-glr-auto-examples-neighbors-plot-nca-dim-reduction-py
    """
    n_neighbors = 3
    random_state = 0

    # Load Digits dataset
    X, y = (embeddings, feature_data["labels"])
    # Split into train/test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=random_state
    )

    dim = len(X[0])
    n_classes = len(np.unique(y))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
    )

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

        # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
        plt.title(
            "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
        )

    plt.show()

    x = embeddings
    y = feature_data["labels"]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="Iris data T-SNE projection")

    knn = NearestNeighbors(n_neighbors=25)
    knn.fit(feature_data["features"])
    neighbours = knn.kneighbors([feature_data["features"][250]], return_distance=False)

    print("Find nearest to:", feature_data["filenames"][250])

    for index in neighbours[0]:
        print(feature_data["filenames"][index])
