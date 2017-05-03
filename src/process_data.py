import sys
import json
from textblob import TextBlob
from collections import defaultdict

POS_TO_EXCLUDE = set()

with open(sys.argv[1]) as data_file:
    data = json.load(data_file)

## Consolidate all the ingredients into one dictionary.
def consolidate_ingredients(data):
    all_ingredients = defaultdict(int)
    for recipe in data:
        for ingredient in recipe['ingredients']:
            all_ingredients[ingredient] += 1
    return all_ingredients
## Reduce ingredient list by passed in flags and output an indexed list and number of ingredients

def save_pos(ingredients):
    pos_map = {}
    for ingredient in ingredients.keys():
        tb = TextBlob(ingredient)
        for word, pos in tb.pos_tags:
            pos_map[word] = pos
    with open(sys.argv[2], 'w') as output:
        json.dump(pos_map, output)

## Create sparse occurrence vector for each recipe
def reduce_ingredients(ingredients):
    with open(sys.argv[1]) as f:
        pos_map = json.load(f)

    reduced = {}
    for ingredient, count in ingredients.iteritems():
        words = ingredient.split()
        n_words_missing = len([word for word in words if word not in pos_map])
        if n_words_missing:
            print 'warning: missing %d pos' % n_words_missing
        reduced[' '.join(word for word in words if pos_map.get(word) not in POS_TO_EXCLUDE)] = count
    return reduced

## test stuff
def print_relevant_stats(ingredients):
    sorted_ingredients = sorted(ingredients.items())

if __name__ == '__main__':
    import time
    ingredients = consolidate_ingredients(data)
    print len(ingredients)
    t = time.time()
    save_pos(ingredients)
    print time.time() - t
    '''
    print 'initial count: ', len(set(ingredients))
    reduced = reduce_ingredients(ingredients)
    print 'reduced count: ', len(set(reduced))
    print '\n'.join(str(x) for x in sorted(reduced.items(), key=lambda p: -p[1])[:100])
    '''
