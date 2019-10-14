
import os
import random
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
                             SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, \
                                BertModel, BertPreTrainedModel, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

from textutils import CompactPreprocessor
from qamodel import QAModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(texts, labels, max_seq_length, tokenizer,
                                 label_map, texts_b=None):
    """
    Loads a data file into a list of `InputBatch`s.
    (adapted from pytorch_pretrained_bert run_classifier.py)
    """

    assert len(texts) == len(labels)
    if texts_b is None:
        # dummy texts
        texts_b = ['' for _ in range(len(texts))]
        has_text_b = False
    else:
        assert len(texts_b) == len(labels)
        has_text_b = True

    features = []
    for example, example_b, label in zip(texts, texts_b, labels):
        tokens_a = tokenizer.tokenize(example)

        tokens_b = None
        if has_text_b:
            tokens_b = tokenizer.tokenize(example_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label]
        features.append([input_ids, input_mask, segment_ids, label_id])
    return features


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def train(args):
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # each time only runs one epoch
    n_epochs = 1
    # current epoch in sequence of calls to this script
    epoch = args.epoch
    learning_rate = args.learning_rate
    warmup_proportion = 0.1

    checkpoint_directory = os.path.join(args.output, 'qamodel')
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    traindata = os.path.join(args.output, 'walkthrough_train_commands_real.csv')
    validdata = os.path.join(args.output, 'walkthrough_valid_commands_real.csv')
    qatrain, qavalid = generate_qa_datasets(traindata, validdata)
    if args.nrows:
        qatrain = qatrain[:args.nrows]
        qavalid = qavalid[:args.nrows]

    model = QAModel.from_pretrained('bert-base-uncased', num_labels=2)
    if epoch > 0:
        checkpoint_name = os.path.join(checkpoint_directory,
                                       'checkpoint_{}.tar'.format(epoch-1))
        checkpoint = torch.load(checkpoint_name, map_location='cpu')
        model.load_state_dict(checkpoint['state'])
    if args.fp16:
        model.half()
    num_train_steps = int(qatrain.shape[0] / batch_size * n_epochs)
    # dummy label map
    label_map = {0:0, 1:1}

    train_features = convert_examples_to_features(qatrain.text.values, qatrain.target.values,
                                    max_seq_length, tokenizer, label_map,
                                    texts_b=qatrain.command.values)

    all_input_ids = torch.tensor([f[0] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f[3] for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    model = model.to(DEVICE)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    if args.fp16:
        optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate,
                            warmup=0.1, t_total=num_train_steps)

    model.train()
    global_step = 0
    predictions = np.array([])
    labels = np.array([])

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits, loss = model(input_ids, segment_ids, input_mask, label_ids)
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if args.fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
        optimizer.step()
        model.zero_grad()
        global_step += 1
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        predictions = np.concatenate([predictions, np.argmax(logits, axis=1)])
        labels = np.concatenate([labels, label_ids])
        if global_step % 1000 == 0:
            acc = metrics.accuracy_score(labels, predictions)
            f1 = metrics.f1_score(labels, predictions)
            print('[{}] Loss {:f} Acc {:f} F1 {:f}'.format(step,
                        tr_loss/nb_tr_steps, acc, f1))

    # Save checkpoint each epoch
    checkpoint_path = os.path.join(checkpoint_directory, 'checkpoint_{}.tar'.format(epoch))
    torch.save({
        'iteration': epoch,
        'state': model.state_dict(),
        'opt': optimizer.state_dict(),
    }, checkpoint_path)
    # the model used by the agent points to this symlink
    symlink_path = os.path.join(checkpoint_directory, 'checkpoint_final.tar')
    if os.path.exists(symlink_path):
        os.unlink(symlink_path)
    os.symlink(checkpoint_path, symlink_path)

    validate(args, qavalid, epoch, model)
    if args.clean and epoch > 0:
        os.remove(os.path.join(checkpoint_directory,
                               'checkpoint_{}.tar'.format(epoch-1)))


def validate(args, qavalid, epoch, model):
    max_seq_length = args.max_seq_length
    num_labels = 2
    # dummy label map
    label_map = {0:0, 1:1}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    valid_features = convert_examples_to_features(qavalid.text.values, qavalid.target.values,
                                    max_seq_length, tokenizer, label_map,
                                    texts_b=qavalid.command.values)

    input_ids = torch.tensor([f[0] for f in valid_features], dtype=torch.long)
    input_mask = torch.tensor([f[1] for f in valid_features], dtype=torch.long)
    segment_ids = torch.tensor([f[2] for f in valid_features], dtype=torch.long)
    label_ids = torch.tensor([f[3] for f in valid_features], dtype=torch.long)
    valid_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size_eval)

    model.eval()

    outputs = np.empty((0, num_labels))
    labels = np.array([])
    cumloss, nsteps = 0, 0
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            logits, loss = model(input_ids, segment_ids, input_mask, label_ids)
            proba = F.softmax(logits, dim=1)
            cumloss += loss.item()
        nsteps += 1
        outputs = np.vstack([outputs, proba.detach().cpu().numpy()])
        labels = np.concatenate([labels, label_ids.detach().cpu().numpy()])

    acc = metrics.accuracy_score(labels, np.argmax(outputs, axis=1))
    f1 = metrics.f1_score(labels, np.argmax(outputs, axis=1))
    print('Validation loss {:f} Acc {:f} F1 {:f}'.format(cumloss/nsteps, acc, f1))

    export_name = os.path.join(args.output, 'qamodel', 'prediction_{}.csv'.format(epoch))
    qavalid['pred'] = labels
    qavalid.to_csv(export_name, index=False)


# Sampling commands from dataset

def make_dataset(df, sample=8, sample2=1):
    irrelevant_commands = ['close', 'drink', 'insert', 'put']
    data = []
    cp = CompactPreprocessor()
    cols = ['entities','inventory','recipe','description','command', 'gamename', 'gen_commands']
    for entities, inventory, recipe, description, command, gamename, gen_commands in tqdm(df[cols].values):
        # add correct command for state
        # FIXME
        #entities = [e for e, _ in row.ext_entities]
        entities = [e.replace('-', ' ') for e in entities]

        text = cp.convert(description, recipe, inventory, entities)
        data.append((text, command, 1, gamename))
        # important commands
        main_commands = [c for c in gen_commands
                         if c != command and c not in irrelevant_commands]
        if sample > 0:
            main_commands = random.sample(main_commands, min(sample, len(main_commands)))
        # add wrong commands for state
        for cmd in main_commands:
            data.append((text, cmd, 0, gamename))
        # irrelevant commands
        bad_commands = [c for c in gen_commands
                         if c != command and c in irrelevant_commands]
        if sample2 > 0:
            bad_commands = random.sample(bad_commands, min(sample2, len(bad_commands)))
        # add wrong commands for state
        for cmd in bad_commands:
            data.append((text, cmd, 0, gamename))

    return pd.DataFrame(data, columns=['text','command', 'target', 'gamename'])


def generate_qa_datasets(traindata, validdata):
    """ generate a dataset for QA model training """

    train = pd.read_csv(traindata)
    valid = pd.read_csv(validdata)
    for c in ['entities', 'gen_commands']:
        train[c] = train[c].apply(eval)
        valid[c] = valid[c].apply(eval)

    qa_valid = make_dataset(valid, sample=5, sample2=1)
    qa_train = make_dataset(train, sample=5, sample2=1)
    return qa_train, qa_valid


def initialize_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help="path for output models")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--batch-size",
                        default=14,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--epoch",
                        required=True,
                        type=int,
                        help="The epoch number being executed")
    parser.add_argument("--learning-rate",
                        default=1e-5,
                        type=float,
                        help="Learning rate for model training")
    parser.add_argument('--nrows',
                        default=0,
                        type=int,
                        help="Number of rows from dataset to use (default 0 uses all data)")
    parser.add_argument("--validate",
                        action='store_true',
                        help="Run validation")
    parser.add_argument("--batch_size_eval",
                        default=156,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--max_seq_length",
                        default=342,
                        type=int,
                        help="Max sequence size in model")
    parser.add_argument("--clean",
                        action='store_true',
                        help="Remove previous model checkpoints")
    args = parser.parse_args()
    initialize_random_seed(args.epoch)
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    train(args)
