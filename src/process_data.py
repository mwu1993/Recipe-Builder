import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import json
#from textblob import TextBlob
from collections import defaultdict
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.mixture import GMM

POS_TO_KEEP = {'NN', 'NNS', 'JJ'}
data_dir = '../data'

def load_data():
    with open(os.path.join(data_dir, 'train.json')) as data_file:
        train_data = json.load(data_file)
    with open(os.path.join(data_dir, 'test.json')) as data_file:
        test_data = json.load(data_file)
    return train_data + test_data

def consolidate_ingredients(data):
    all_ingredients = defaultdict(int)
    for recipe in data:
        for ingredient in recipe['ingredients']:
            all_ingredients[ingredient] += 1
    return all_ingredients

def save_pos(ingredients, out_filepath):
    pos_map = {}
    for ingredient in ingredients:
        tb = TextBlob(ingredient)
        for word, pos in tb.pos_tags:
            pos_map[word] = pos
    with open(out_filepath, 'w') as output:
        json.dump(pos_map, output)

def simplify_ingredient_name(ingredient, count, last_words, pos_map):
    words = [word.lower() for word in ingredient.replace(',', '').split()]
    last_word = words[-1]
    if last_word[-1] == 's' and last_word[:-1] in last_words:
        last_word = last_word[:-1]
    if count > 500:
        return ' '.join(
            word for word in words[:-1] if pos_map.get(word) in POS_TO_KEEP
        ) + last_word
    else:
        return last_word

def reduce_ingredients(ingredients, pos_map):
    last_words = set([ingredient.split()[-1] for ingredient in ingredients])
    reduced = defaultdict(int)

    for ingredient, count in ingredients.iteritems():
        new_ingredient = simplify_ingredient_name(ingredient, count, last_words, pos_map)
        reduced[new_ingredient] += count

    return {ingredient: count for ingredient, count in reduced.iteritems() if count > 50}, last_words

def get_recipe_ingredient_matrix(data, pos_map):
    ingredients = consolidate_ingredients(data)
    reduced_ingredients, last_words = reduce_ingredients(ingredients, pos_map)
    ingredient_indices = dict(zip(reduced_ingredients, range(len(reduced_ingredients))))
    ingredient_order = dict((i, ing) for ing, i in ingredient_indices.iteritems())
    matrix = dok_matrix((len(data), len(ingredient_indices)), dtype=np.float32)
    recipe_id_to_cuisine = {}

    for i, recipe in enumerate(data):
        recipe_id_to_cuisine[i] = recipe.get('cuisine')
        for ingredient in recipe['ingredients']:
            ingredient_name = simplify_ingredient_name(
                ingredient, ingredients[ingredient], last_words, pos_map
            )
            if ingredient_name in ingredient_indices:
                matrix[i, ingredient_indices[ingredient_name]] = 1./np.log(
                    reduced_ingredients[ingredient_name])

    print matrix.sum(), matrix.shape
    svd = TruncatedSVD(n_components=200)
    recipe_vectors = svd.fit_transform(matrix)
    print sum(svd.explained_variance_ratio_)

    gmm = GMM(n_components=3, covariance_type='diag')
    gmm.fit(recipe_vectors)
    score = gmm.score(recipe_vectors)
    print len(score), sum(score)

    sampled_vectors = gmm.sample(100)
    sampled_ingredients = svd.inverse_transform(sampled_vectors)
    for row in sampled_ingredients:
        print [ingredient_order[i] for i in range(len(row)) if row[i] > .7/np.log(
            reduced_ingredients[ingredient_order[i]])]

def visualize_cuisines(recipe_vectors, recipe_cuisines, n_samples=1000):
    cuisine_counts = defaultdict(int)
    for cuisine in recipe_cuisines.itervalues():
        cuisine_counts[cuisine] += 1
    top_cuisines = set([
        cuisine
        for cuisine, count in
        sorted(cuisine_counts.items(), key=lambda pair: pair[1], reverse=True)[:7]
        if cuisine is not None
    ])
    color_map = dict(zip(top_cuisines, ('yellow', 'green', 'blue', 'black', 'red', 'pink')))
    print color_map
    recipe_indices = random.sample([
        i for i, cuisine in recipe_cuisines.iteritems() if cuisine in top_cuisines
    ], n_samples)
    recipe_vectors = recipe_vectors[recipe_indices, :]

    model = TSNE(n_components=2)
    recipe_vectors_visualization = model.fit_transform(recipe_vectors)
    plt.scatter(
        recipe_vectors_visualization[:,0],
        recipe_vectors_visualization[:,1],
        color=[color_map[recipe_cuisines[i]] for i in recipe_indices],
    )
    plt.show()
    #plt.plot([svd.explained_variance_ratio_[:k].sum() for k in range(1, 501)])
    #plt.show()

def run():
    data = load_data()
    print len(data)
    ingredients = consolidate_ingredients(data)
    with open(os.path.join(data_dir, 'train_ingredient_pos.json')) as f:
        pos_map = json.load(f)

    print 'initial count: ', len(ingredients)
    reduced = reduce_ingredients(ingredients, pos_map)
    print 'reduced count: ', len(reduced)
    for ingredient, count in sorted(reduced.items(), key=lambda p: -p[1]):
        print ingredient, count

    return ingredients, reduced

if __name__ == '__main__':
    data = load_data()
    with open(os.path.join(data_dir, 'train_ingredient_pos.json')) as f:
        pos_map = json.load(f)
    get_recipe_ingredient_matrix(data, pos_map)

