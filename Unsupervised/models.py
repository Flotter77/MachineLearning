from nltk import pos_tag
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF, FactorAnalysis
from sklearn.cluster import KMeans, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords, wordnet
from collections import Counter
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

# from fancyimpute import MICE as MICE
from impyute.imputation.cs import mice
import pickle
import csv
import operator
# import datawig

#  https://python.gotrained.com/text-classification-with-pandas-scikit/ for basic bags-of-words
from pickle_func import pickle_dump, pickle_load

stop_words_extra = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                    'should', 'now', 'uses', 'use', 'using', 'used', 'one', 'also', 'br', 'href', 'ilink', 'whether']
stop_words = stopwords.words("english")
stop_words = stop_words + stop_words_extra

# Preporcess data 
def preprocess(data):
    reviews_tokens = []
    for review in data:
        review = review.lower()  # Convert to lower-case words
        raw_word_tokens = re.findall(r'(?:\w+)', review, flags=re.UNICODE)  # remove pontuaction
        word_tokens = [w for w in raw_word_tokens if not w in stop_words]  # do not add stop words
        reviews_tokens.append(word_tokens)
    return reviews_tokens  # return all tokens


# def construct_bag_of_words(data):
#     corpus = preprocess(data)
#     bag_of_words = {}
#     word_count = 0
#     for sentence in corpus:
#         for word in sentence:
#             if word not in bag_of_words:  # do not allow repetitions
#                 bag_of_words[word] = word_count  # set indexes
#                 word_count += 1
#     print(dict(Counter(bag_of_words).most_common(5)))
#     return bag_of_words  # index of letters

def construct_bag_of_words_freq(data):
    corpus = preprocess(data)
    bag_of_words = {}
    for sentence in corpus:
        for word in sentence:
            if word not in bag_of_words:  # do not allow repetitions
                bag_of_words[word] = 1  # set indexes
            else:
                bag_of_words[word] = bag_of_words[word] + 1

    bag_of_words_thres = {key: val for key, val in bag_of_words.items() if val > 20}
    print("bag of word counts (filtered): ")
    print(dict(Counter(bag_of_words_thres).most_common(20)))
    # bag_of_words_noun = {key: val for key, val in bag_of_words_thres.items() if pos_tag([key])[0][1] == 'NN'}
    # print("bag of word counts (noun): ")
    # print(dict(Counter(bag_of_words_noun).most_common(20)))
    # bag_of_words_adj = {key: val for key, val in bag_of_words_thres.items() if pos_tag([key])[0][1] == 'JJ'}
    # print("bag of word counts (adjective): ")
    # print(dict(Counter(bag_of_words_adj).most_common(20)))

    bag_of_words_out = dict(Counter(bag_of_words_thres).most_common(2000))
    # ---------------- CHANGE TO bag_of_words_adj IF YOU WANT ADJECTIVE ----------------#
    index = 0
    for key, val in bag_of_words_out.items():
        bag_of_words_out[key] = index
        index += 1

    # bag_of_words_sorted = sorted(bag_of_words_noun.items(), key=operator.itemgetter(1))
    # convert back from freq to index
    return bag_of_words_out


def featurize(sentence_tokens, bag_of_words):
    sentence_features = [0 for x in range(len(bag_of_words))]
    for word in sentence_tokens:
        if word in bag_of_words.keys():
            index = bag_of_words[word]
            sentence_features[index] += 1
    return sentence_features


def get_batch_features(data, bag_of_words):
    batch_features = []
    reviews_text_tokens = preprocess(data)
    for review_text in reviews_text_tokens:
        feature_review_text = featurize(review_text, bag_of_words)
        batch_features.append(feature_review_text)
    return batch_features


# profile_df = pd.read_csv("profiles.csv")
# profile_df.head()
# print('Step 0. num_users: %d  num_features: %d' % (profile_df.shape[0], profile_df.shape[1]))
#
# # Remove -1 (rather not say) in the income column and remove the last online column
# # profile_df = profile_df[profile_df['income'] != -1]
# profile_df = profile_df.drop(['last_online'], axis=1)
# print('Step 1. num_users: %d  num_features: %d' % (profile_df.shape[0], profile_df.shape[1]))
#
# # One-hot encoding to convert the categorical data into the binary values
# categorical_columns = ['body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity', 'job', 'offspring',
#                        'orientation', 'pets', 'religion', 'sex', 'sign', 'smokes', 'speaks', 'status', 'location']
# essay_columns = ['essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9']
# numerical_columns = ['age', 'height', 'income']
# # profile_num_df = pd.get_dummies(profile_df, columns=categorical_columns)
# # profile_num_df2 = pd.get_dummies(profile_df, columns=numerical_columns)
# # print('Step 2. num_users: %d  num_features: %d (one-hot encoding: categorical)' % (profile_num_df.shape[0], profile_num_df.shape[1]))
# # print('Step 2. num_users: %d  num_features: %d (one-hot encoding: numerical)' % (profile_num_df2.shape[0], profile_num_df2.shape[1]))
#
# # Get the columns that are only numerical (no essay)
# # num_columns = [c for c in profile_num_df.columns.values if (profile_num_df[c].dtype == np.uint8) or
# #                (profile_num_df[c].dtype == np.int64) or (profile_num_df[c].dtype == np.float64)]
# # Divide the dataset according to the 3 type: numerical, categorical, essay (feed in separately?)
# # profile_categorical = profile_num_df.drop(['age', 'height', 'income'], axis=1)
# profile_categorical = pd.get_dummies(profile_df[categorical_columns], columns=categorical_columns)
# profile_numerical = pd.get_dummies(profile_df[numerical_columns], columns=numerical_columns)
# profile_essay = profile_df[essay_columns]
#
# print('Step 3. num_users: %d  num_features: %d (numerical)' % (profile_numerical.shape[0], profile_numerical.shape[1]))
# print('Step 3. num_users: %d  num_features: %d (categorical)' % (profile_categorical.shape[0], profile_categorical.shape[1]))
#
# # Bags-of-words for essay features
# profile_essay = profile_essay.replace(np.nan, '', regex=True)  # remove NaN
# profile_essay = profile_essay[essay_columns].apply(lambda x: ' '.join(x), axis=1)  # concatenate essays into paragraph
# profile_essay = profile_essay.str.replace('\d+', '')  # remove digits in the paragraph
#
# bag_of_words = construct_bag_of_words_freq(profile_essay)
# profile_essay = get_batch_features(profile_essay, bag_of_words)
# profile_essay = np.asarray(profile_essay).astype(float)
# profile_essay_df = pd.DataFrame(data=profile_essay, columns=list(bag_of_words.keys()))
# print('Step 4. num_users: %d  num_features: %d (essay)' % (profile_essay.shape[0], profile_essay.shape[1]))
#
# profile = pd.concat([profile_numerical, profile_categorical], axis=1)
# profile.drop([col for col, val in profile.sum().iteritems() if val < 2], axis=1, inplace=True)
# profile = pd.concat([profile, profile_essay_df], axis=1)
#
# # profile_essay_1000 = []
# # for essay_idx in essay_columns:
# #     essay = profile_essay[essay_idx]
# #     essay = essay.str.replace('\d+', '')  # remove digits in the paragraph
# #     bag_of_words = construct_bag_of_words_freq(essay)
# #     data = get_batch_features(essay, bag_of_words)
# #     data = np.asarray(data).astype(float)
# #     profile_essay_1000.append(pd.DataFrame(data=data, columns=list(bag_of_words.keys())))
# # profile_essay = pd.concat(profile_essay_1000, axis=1)
# # print('Step 3. num_users: %d  num_features: %d (essay)' % (profile_essay.shape[0], profile_essay.shape[1]))
#
# # print('Step 4. num_users: %d  num_features: %d (essay with noun)' % (profile_essay.shape[0], profile_essay.shape[1]))
#
#
# train, test = train_test_split(profile, test_size=0.2)
#
# pickle_dump(train, "train_new.pkl")
# pickle_dump(test, "test_new.pkl")

# Impute MNAR via finding the features correlated to income (https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.6902)
# 0 = observed,  1 = missing
# profile_income_df = profile_df['income'].mask(profile_df['income'] != -1, 0)
# profile_income_df = profile_income_df.mask(profile_income_df == -1, 1)
# profile_new = pd.concat([profile_numerical, profile_categorical, profile_essay], axis=1)
# profile_new = profile_new.drop('income', axis=1).apply(lambda x: x.corr(profile_income_df))
#
# # METHOD - MICE
# profile_new['income'] = profile_new['income'].replace(-1, np.nan, regex=True)  # remove NaN
# imputed_training = mice(profile_new.values)

# # METHOD - DEEP LEARNING
# # part 1) find the features that contain information about the target feature that will be imputed
# # part 2) initialize a SimpleImputer model
# imputer = datawig.SimpleImputer(
#     input_columns=['age', '', '3'],  # column(s) containing information about the column we want to impute
#     output_column='income',  # the column we'd like to impute values for
#     output_path='imputer_model'  # stores model data and metrics
#     )
# # part 3) fit an imputer model on the train data
# imputer.fit(train_df=profile_new[profile_new['income'] != -1], num_epochs=50)
# # part 4) impute missing values and return original dataframe with predictions
# imputed = imputer.predict(profile_new[profile_new['income'] == -1])
# with open('profile_essay.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(profile_essay)
#
# train = pickle_load("train.pkl")
# test = pickle_load("test.pkl")
#
# train = train.iloc[:, :127+8293]
# test = test.iloc[:, :127+8293]
#
# # profile = pd.concat([train, test], axis=0)
# profile.drop([col for col, val in profile.sum().iteritems() if val < 2], axis=1, inplace=True)
# print('Step 4. num_users: %d  num_features: %d' % (profile.shape[0], profile.shape[1]))
# # np.isnan(profile).any().any()
# train, test = train_test_split(profile, test_size=0.2)

train = pickle_load("train.pkl")
test = pickle_load("test.pkl")

models = [
    {
        'name': 'PCA',
        'function': PCA(n_components=5)
    },
    {
        'name': 'LDA',
        'function': LatentDirichletAllocation(n_components=5)
    },
    {
        'name': 'FactorAnalysis',
        'function': FactorAnalysis(n_components=5)
    },
    {
        'name': 'GaussianMixture',
        'function': GaussianMixture(n_components=5)
    },
    {
        'name': 'NMF',
        'function': NMF(n_components=5)
    }
]

clusters = [
    {
        'name': 'KMeans',
        'function': KMeans()
    }
]

for n in models:
    n.update({'train_time': 0})
    n.update({'train_RE': 0})
    n.update({'test_RE': 0})
    n.update({'train_LL': 0})
    n.update({'test_LL': 0})
    n.update({'train_ALL': 0})
    n.update({'test_ALL': 0})
    n.update({'train_aic': 0})
    n.update({'test_aic': 0})
    n.update({'train_bic': 0})
    n.update({'test_bic': 0})

    print('\n -----' + n['name'] + '-----')

    model = n['function']

    t0 = time()
    model.fit(train.values)
    n['train_time'] = time() - t0
    print("train time: %0.2fs" % n['train_time'])

    latent_users = model.components_
    train_transformed = model.transform(train.values)  # "loadings" for each sample, meaning how much of each component you need to describe it best using a linear combination of the components_ (the principal axes in feature space).
    test_transformed = model.transform(test.values)

    print("Latent users shape:", latent_users.shape)
    print("Train - User proportions from latent users shape: ", train_transformed.shape)
    print("Test - User proportions from latent users shape: ", test_transformed.shape)

    if not n['name'] == 'NMF':
        n['train_LL'] = model.score(train.values)
        n['test_LL'] = model.score(test.values)
        print("Train - Log-likelihood: ", n['train_LL'])
        print("Test - Log-likelihood: ", n['test_LL'])
        n['train_ALL'] = model.score(train.values) / train.shape[0]  # not working when num_samp < num_feature
        n['test_ALL'] = model.score(test.values) / test.shape[0]
        print("Train - Average Log-Likelihood: ", n['train_ALL'])
        print("Test - Average Log-Likelihood: ", n['test_ALL'])

    if n['name'] == 'PCA' or n['name'] == 'NMF':
        train_origin = model.inverse_transform(train_transformed)
        test_origin = model.inverse_transform(test_transformed)

        n['train_RE'] = ((train.values - train_origin) ** 2).mean()
        n['test_RE'] = ((test.values - test_origin) ** 2).mean()
        print("Train - Reconstruction Error: ", n['train_RE'])
        print("Test - Reconstruction Error: ", n['test_RE'])

    if n['name'] == 'GaussianMixture':
        n['train_aic'] = model.aic(train.values)
        n['test_aic'] = model.aic(test.values)
        n['train_bic'] = model.bic(train.values)
        n['test_bic'] = model.bic(test.values)

    # plt.figure()
    # for i in range(train_transformed.shape[1]):
    #     plt.hist(train_transformed[:, i], alpha=0.3, label="Latent User " + str(i + 1), range=(0, 1), bins=20)
    # plt.xlabel("User Proportion from Latent User i", fontsize=20)
    # plt.ylabel("Count", fontsize=20)
    # plt.tick_params(labelsize=15)
    # plt.legend()
    # plt.show(block=False)

    plt.figure()
    targets = [1, 0]
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = train['sex_m'] == target
        plt.scatter(train_transformed[indicesToKeep, 0], train_transformed[indicesToKeep, 1], c=color)
    plt.lengend(targets)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show(block=False)

    if n['name'] == 'PCA':
        kmeans = KMeans().fit(train_transformed)  # n_clusters=2
        # cluster_transformed = kmeans.transform()
        centroids = kmeans.cluster_centers_
        print("Train - Average Log-Likelihood with Kmeans: ", kmeans.score(train_transformed) / train.shape[0])
        print("Test - Average Log-Likelihood with Kmeans: ", kmeans.score(test_transformed) / test.shape[0])
        plt.figure()
        plt.plot(train_transformed[:, 0], train_transformed[:, 1], 'k.', markersize=2)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title('K-means clustering on the dataset (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.show(block=False)

hrange = np.arange(1, 20, 2)
hacc = np.zeros(len(hrange))
hacc2 = np.zeros(len(hrange))

for idx in range(len(hrange)):
    model = PCA(n_components=hacc[idx])
    model.fit(train.values)
    train_transformed = model.transform(train.values)
    hacc[idx] = model.score(train.values) / train.shape[0]
    train_origin = model.inverse_transform(train_transformed)
    hacc2[idx] = ((train.values - train_origin) ** 2).mean()
    plt.figure()
    plt.scatter(train_transformed[:, 0], train_transformed[:, 1])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show(block=False)

plt.figure()
plt.plot(hrange, hacc, label="LL")
plt.plot(hrange, hacc2, label="RE")
plt.xlabel('The number of components')
plt.ylabel('Error')
plt.ylim([0.0, .30])
plt.legend(loc="lower right")
# plt.title('Neural Network with varying hidden layer size')
plt.show(block=False)
