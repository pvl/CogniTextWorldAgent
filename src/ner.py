import os
from bertner import Ner
from textutils import CompactPreprocessor

BASEPATH = os.path.dirname(os.path.abspath(__file__))
MODEL = None


def extract_entities(description, inventory, model=None):
    global MODEL
    if model is None:
        if MODEL is None:
            modelpath = os.path.join(os.path.dirname(BASEPATH),
                                     'outputs', 'nermodel')
            MODEL = Ner(modelpath)
        model = MODEL

    cp = CompactPreprocessor()
    state = cp.convert(description, '', inventory, [])

    pred = model.predict(state.lower())
    entities_types = extract_entity_list(pred)
    return entities_types


def extract_entity_list(predictions):
    res = []
    in_progress = ''
    etyp = ''
    for word in predictions:
        if word['tag'] == 'O':
            if in_progress:
                res.append((in_progress.strip(), etyp))
                in_progress, etyp = '', ''
        elif word['tag'].startswith('B-'):
            if in_progress:
                res.append((in_progress.strip(), etyp))
            in_progress = word['word']
            etyp = word['tag'][2:]
        elif word['tag'].startswith('I-'):
            in_progress += ' ' + word['word']
            etyp = word['tag'][2:]
    if in_progress:
        res.append((in_progress.strip(), etyp))
    return list(set(res))
