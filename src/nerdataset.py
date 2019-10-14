import os
import random
import argparse
import pandas as pd
from textutils import CompactPreprocessor

categories = {
    'D': ['plain door', 'sliding patio door', 'front door',
          'commercial glass door', 'sliding door', 'patio door',
          'fiberglass door', 'barn door', 'screen door',
          'wooden door', 'frosted glass door'],
    'C': ['fridge', 'toolbox'],
    'W': ['south', 'west', 'east', 'north'],
    'S': ['counter', 'toilet', 'table', 'showcase', 'workbench', 'bed',
          'sofa', 'shelf', 'patio chair', 'oven', 'BBQ', 'stove',
          'patio table'],
    'T': ['knife', 'cookbook'],
    'F': ['orange bell pepper', 'olive oil', 'purple potato', 'banana',
          'flour', 'chicken wing', 'water', 'parsley', 'yellow bell pepper',
          'red potato', 'patio table', 'yellow potato', 'chicken leg',
          'block of cheese', 'cilantro', 'meal', 'salt', 'carrot',
          'white onion', 'red hot pepper', 'black pepper', 'red apple',
          'pork chop', 'red onion'],
}

AUGMENT = {
    'D': ['metal door', 'gate', 'hatch', 'flush door', 'aluminum door',
          'plastic door', 'pocket door', 'roller door', 'pivot door'],
    'C': ['microwave', 'drawer', 'closet', 'package', 'oven', 'box',
          'bookcase', 'wardrobe'],
    'S': ['desk', 'sideboard', 'armchair', 'lift chair', 'mattress',
          'computer desk'],
    'T': ['scissors', 'scissors', 'blue screwdriver', 'green screwdriver',
          'spoon', 'large ratchet', 'small ratchet', 'pliers',
          'magazine', 'newspaper', 'pencil'],
    'F': ['orange', 'tomato', 'lettuce', 'pumpkin', 'pineapple',
          'peach', 'lemon', 'sprite melon', 'water melon', 'steak', 'pizza',
          'sliced bread', 'burger', 'croissant', 'strawberries', 'fish',
          'ice cream']
}


def get_game_entities():
    return [v for vls in categories.values() for v in vls]


def get_all_entities():
    game_ents = get_game_entities()
    augment_ents = [v for vls in AUGMENT.values() for v in vls]
    return game_ents + augment_ents


REVCAT = None


def get_category(entity):
    global REVCAT
    if not REVCAT:
        REVCAT = {}
        for c, values in categories.items():
            for v in values:
                REVCAT[v] = c
        for c, values in AUGMENT.items():
            for v in values:
                REVCAT[v] = c
    return REVCAT.get(entity)


def alternate_entity(entity):
    cat = get_category(entity)
    if cat and cat in AUGMENT:
        return random.choice(AUGMENT[cat])
    return entity


def make_entity(parts):
    return ' '.join(parts)


def generate_candidates(token, bctx, actx, size=3):
    out = [token]
    for i in range(1, size+1):
        out.append(make_entity([token]+actx[:i]))
        out.append(make_entity(bctx[-i:] + [token]))
        out.append(make_entity(bctx[-i+1:] + [token] + actx[:i-1]))
    return out


def entity_type(token, bctx, actx, entities):
    candidates = generate_candidates(token, bctx, actx)
    matches = [c for c in candidates if c in entities]
    if matches:
        return get_category(matches[0])
    return "O"


def generate_bio_tokens(txt, entities):
    tokens = txt.lower().split()
    res = []
    prev_tag = 'O'
    for i, token in enumerate(tokens):
        tag = entity_type(token, tokens[i-3:i], tokens[i+1:i+4], entities)
        if tag == 'O':
            res.append((token, tag))
            prev_tag = tag
        elif tag != 'O' and prev_tag == 'O':
            res.append((token, 'B-' + tag))
            prev_tag = tag
        elif prev_tag != 'O' and prev_tag == tag:
            res.append((token, 'I-' + tag))
            prev_tag = tag
        elif prev_tag != 'O' and prev_tag != tag:
            res.append((token, 'B-' + tag))
            prev_tag = tag
    return res


def augment_text(txt, proba=0.1):
    for ent in get_game_entities():
        if ent in txt and random.random() < proba:
            txt = txt.replace(ent, alternate_entity(ent))
    return txt


def make_dataset(dd, name):
    cp = CompactPreprocessor()
    output = ['-DOCSTART- -X- -X- O', '']
    for _, s in dd.iterrows():
        txt = cp.convert(s.description, '', s.inventory, s.entities)
        txt = augment_text(txt)
        entities = get_all_entities()
        btk = generate_bio_tokens(txt, entities)
        # add dummy entries
        btk = [(t, 'X', 'X', v) for t, v in btk]
        output += [' '.join(b) for b in btk]
        output.append('')

    with open(name, 'w') as f:
        f.write('\n'.join(output))


def generate(basepath):

    wt1 = pd.read_csv(os.path.join(basepath, 'walkthrough_valid_cookbook.csv'))
    wt2 = pd.read_csv(os.path.join(basepath, 'walkthrough_train_cookbook.csv'))
    wt = pd.concat([wt1, wt2])
    for c in ['entities', 'admissible_commands']:
        wt[c] = wt[c].apply(eval)

    wt = wt.sample(len(wt))
    tr = wt.iloc[:40000]
    te = wt.iloc[40000:50000]
    va = wt.iloc[50000:]

    if not os.path.exists(os.path.join(basepath, 'nerdata')):
        os.makedirs(os.path.join(basepath, 'nerdata'))

    make_dataset(tr, os.path.join(basepath, 'nerdata', 'train.txt'))
    make_dataset(va, os.path.join(basepath, 'nerdata', 'valid.txt'))
    make_dataset(te, os.path.join(basepath, 'nerdata', 'test.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Generate datasets for NER model training")
    parser.add_argument('--output', type=str, default='../datafiles',
                        help="path to processed datasets")
    args = parser.parse_args()
    generate(args.output)
