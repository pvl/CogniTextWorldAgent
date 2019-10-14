from typing import List, Dict, Any, Optional
from textworld import EnvInfos
from collections import Counter, namedtuple
import re
import os
import numpy as np
from scipy.special import softmax
import torch
from torch import nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import logging

from qamodel import QAModel, bprocess
from bertner import Ner
from commandgenerator import CommandModel
from ner import extract_entities
from textutils import CompactPreprocessor, ConnectionGraph, Connection

DEBUG = False
DEVICE = "cuda"

if DEBUG:
    logging.basicConfig(filename='gameplay.log', filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger('agent')
dbg = logger.debug

State = namedtuple('State','description inventory recipe')

TMF = "26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084"
BMF = "9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba"


def text_preprocess(state, entities):
    pp = CompactPreprocessor()
    return pp.convert(state.description, state.recipe, state.inventory, entities)


def maxmin_norm(p):
    return (p - p.min())/(p.max() - p.min())


def ucb1(action_cnt, total_cnt):
    if action_cnt == 0:
        return 5
    else:
        return np.sqrt(2*np.log(total_cnt)/action_cnt)


def choose_action(logits, sacnt, alpha=1):
    """
    :param logits: vector with logits for actions
    :param sacnt: vector with counts for each visit of the action
    :returns: action number
    """
    total_visits = sum(sacnt)
    uscore = [alpha*ucb1(v, total_visits) for v in sacnt]
    ssc = maxmin_norm(logits) + uscore
    return np.argmax(ssc), softmax(ssc)


class BaseAgent:
    """
    This agent uses a question and answer model to get probabilities
    for different actions and then chooses a command using probabilities
    and UCB1 to explore on multiple visits to the same state.
    """

    def __init__(self, extra_args=None) -> None:
        self.max_seq_length = 342
        self.device = DEVICE
        self.eval_batch_size = 50
        if extra_args and 'docker' in extra_args:
            self.docker = extra_args['docker']
        else:
            self.docker = True

        if self.docker:
            fn = os.path.join('/root/.pytorch_pretrained_bert', BMF)
            self.model = QAModel.from_pretrained(fn, cache_dir=None, num_labels=2)
            fn = os.path.join('/root/.pytorch_pretrained_bert', TMF)
            self.tokenizer = BertTokenizer.from_pretrained(fn, cache_dir=None)
            self.ner_datapath = '/root/nercheckpoint'
        else:
            self.model = QAModel.from_pretrained('bert-base-uncased', num_labels=2)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.ner_datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             'outputs', 'nermodel')

        self.ner = Ner(self.ner_datapath, docker=self.docker)
        self.langmodel = CommandModel()
        self.load_model()
        self.state_action_list = None
        self.state_action_cache = {}
        # keep the previous state-action
        self.previous_state_action = None
        self.state_action_danger = {}
        # capture the recipe when agent examines the cookbook
        self.recipe = None
        self.previous_action = None
        self.n_model_evals = 0
        # map of places with list of action_spaces
        self.worldmap = ConnectionGraph()
        # regex for hifen words workaround
        self.rgx = re.compile(r'\b(\w+\-\w+)\b')
        self.hifen_map = {}
        self.hifen_rev_map = {}

    def load_model(self):
        if self.docker:
            checkpoint = torch.load('/root/qa_checkpoint.tar', map_location=self.device)
        else:
            fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'outputs', 'qamodel', 'checkpoint_final.tar')
            checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['state'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def select_additional_infos(self) -> EnvInfos:
        raise NotImplementedError

    def init_state_action_list(self, n_games):
        """ list with state-action counter for each game """
        if self.state_action_list is None:
            self.state_action_list = [Counter() for _ in range(n_games)]
            self.previous_state_action = [None for _ in range(n_games)]
            self.recipe = ['' for _ in range(n_games)]
            self.previous_action = ['' for _ in range(n_games)]

    def short_entities(self, entities):
        """ return last word from the entities if there are no duplicats
        otherwise keep the full text
        """
        entities = list(set(entities))
        short = [(e, e.split(' ')[-1]) for e in entities]
        cnt = Counter([s[1] for s in short])
        res = []
        for ent, short_ent in short:
            if cnt[short_ent] == 1:
                res.append(short_ent)
            else:
                res.append(ent)
        return res

    def get_location(self, description):
        return description.splitlines()[0].strip()

    def generate_entities(self, infos, recipe):
        raise NotImplementedError

    def generate_commands(self, infos, recipe, entities):
        raise NotImplementedError

    def preprocess_description(self, description):
        mobj = self.rgx.search(description)
        if mobj:
            kw = mobj.group(0)
            target = kw.replace('-', ' ')
            self.hifen_map[kw] = target
            self.hifen_rev_map[target] = kw
            return description.replace(kw, target)
        return description

    def predict_command(self, infos, recipe):
        description = self.preprocess_description(infos['description'])
        s = State(description=description, inventory=infos['inventory'], recipe=recipe)
        entity_types = self.generate_entities(infos, recipe)
        entities = [e for e,_ in entity_types]
        commands = self.generate_commands(infos, recipe, entity_types)

        if s in self.state_action_cache:
            cmds, pred = self.state_action_cache[s]
            return s, cmds, pred

        text = text_preprocess(s, entities)
        evaldata = bprocess([text]*len(commands),
                            self.max_seq_length, commands, self.tokenizer)
        sampler = SequentialSampler(evaldata)
        dataloader = DataLoader(evaldata, sampler=sampler,
                                batch_size=self.eval_batch_size)
        results = []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch

            with torch.no_grad():
                pred = self.model(input_ids, segment_ids, input_mask)
                pred = pred[:,1].detach().cpu().numpy()
                results.append(pred)
                self.n_model_evals += 1
        results = np.concatenate(results) if len(results) > 0 else np.array(results)

        top_commands = list(reversed(sorted(zip(commands, results), key=lambda x: x[1])))
        commands = [v[0] for v in top_commands]
        results = np.array([v[1] for v in top_commands])

        self.state_action_cache[s] = (commands, results)
        dbg('State: {}'.format(text))
        dbg('Predicted: ' + ','.join(
                '({}, {:5.2f})'.format(c,p) for c,p in reversed(
                    sorted(zip(commands, results), key=lambda x: x[1]))))
        return s, commands, results

    def get_action_cnt(self, commands, state, state_action_cnt):
        res = []
        # check how many go commands exist, if only one it should not be penalized
        # the return to the same location
        n_go_commands = len([c for c in commands if c.startswith('go')])
        location = self.get_location(state.description)
        known_exits = self.worldmap.known_exits(location)
        # all exits were explored restart the penalties to allow agent to return
        if n_go_commands == len(known_exits):
            self.worldmap.reset_except_last(location)
            known_exits = self.worldmap.known_exits(location)
        
        dbg('[Known Exits] {} -> {}'.format(location, known_exits))
        for action in commands:
            sac = state_action_cnt.get((state,action), 0)
            
            # reset counter for go commands
            if action.startswith('go'):
                sac = 0
            # put a +1 penalty to avoid returning to the origin location
            if action in known_exits and n_go_commands > len(known_exits):
                sac = 1
                
            # set a large count (penalty) if the state action shows
            # as a critical danger
            if self.state_action_danger.get((state,action)):
                sac = 1000
            
            res.append(sac)
        return res

    def format_action_list(self, pred, action_cnt):
        """
        Select the t
        """
        pass

    def act_single(self, infos, state_action_cnt, recipe, prev_state_action):
        self.update_map(infos['description'], prev_state_action)
        state, commands, pred = self.predict_command(infos, recipe)
        action_cnt = self.get_action_cnt(commands, state, state_action_cnt)
        dbg('[Action Count] {}'.format(action_cnt))
        action_idx, action_proba = choose_action(pred, action_cnt)
        action = commands[action_idx]
        # needs to increase the action_list counter
        state_action_cnt[(state, action)] += 1
        return state, action, zip(commands, action_proba)

    def act_detail(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[List[str]]:
        """ Acts upon the current list of observations. 
        Returns both actions and actions probabilities.
        """
        self.init_state_action_list(len(obs))

        self.process_obs(obs)

        actions = []
        actions_proba = []
        for idx in range(len(obs)):
            # skip calculation for games that ended
            if dones[idx]:
                action = ''
                cmdproba = []
                if infos['has_lost'][idx]:
                    # player died save the last state action as a danger
                    if self.previous_state_action[idx]:
                        self.state_action_danger[self.previous_state_action[idx]] = True
                # game ended - reset the counter
                self.state_action_list[idx] = Counter()
                self.previous_state_action[idx] = None
                self.worldmap = ConnectionGraph()
            else:
                ninfo = {k:infos[k][idx] for k in infos}
                state, action, cmdproba = self.act_single(ninfo, 
                                                self.state_action_list[idx], 
                                                self.recipe[idx],
                                                self.previous_state_action[idx])

                self.previous_state_action[idx] = (state, action)
            actions.append(action)
            actions_proba.append(cmdproba)

        dbg('[Selected action] {}'.format(actions))
        self.previous_action = actions
        return actions, actions_proba

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[List[str]]:
        """ Acts upon the current list of observations. Returns the actions """
        actions, _ = self.act_detail(obs, scores, dones, infos)
        return actions

    def update_map(self, description, previous_state_action):
        prev_state, prev_action = previous_state_action if previous_state_action else ('', '')
        if prev_state:
            prev_loc = self.get_location(prev_state.description)
            loc = self.get_location(description)
            if loc != prev_loc:
                if prev_action.startswith('go'):
                    connection = Connection(prev_loc, prev_action, loc)
                    self.worldmap.add(connection)
                else:
                    dbg('[ERROR add connection] {} {} {}'.format(prev_loc, prev_action, loc))


    def process_obs(self, observations):
        # update recipe if last command was examine cookbook
        dbg('Observations: {}'.format(observations))
        for idx, (obs, prevcmd) in enumerate(zip(observations, self.previous_action)):
            if prevcmd == 'examine cookbook':
                self.recipe[idx] = self.clean_recipe(obs)
            dbg('[OBS for {}] {}'.format(prevcmd, obs))

    def clean_recipe(self, recipe):
        return recipe.replace('You open the copy of "Cooking: A Modern Approach (3rd Ed.)" and start reading:','')


class Agent(BaseAgent):
    def select_additional_infos(self) -> EnvInfos:
        request_infos = EnvInfos()
        request_infos.has_won = True
        request_infos.has_lost = True
        request_infos.description = True
        request_infos.inventory = True
        request_infos.command_templates = True
        request_infos.entities = False
        request_infos.admissible_commands = False
        return request_infos

    def entities_mapping(self, entities):
        res = []
        for e,t in entities:
            for k in self.hifen_rev_map.keys():
                if k in e:
                    e = e.replace(k, self.hifen_rev_map[k])
            res.append((e,t))
        return res
    
    def generate_entities(self, infos, recipe):
        description = self.preprocess_description(infos['description'])
        entities = extract_entities(description, infos['inventory'], model=self.ner)
        return self.entities_mapping(entities)

    def generate_commands(self, infos, recipe, entities):
        description = self.preprocess_description(infos['description'])
        commands = self.get_admissible_commands(description, infos['inventory'],
                                                entities, infos['command_templates'])
        return commands

    def get_admissible_commands(self, description, inventory, entities, templates):
        state_entities = entities
        dbg('State entities: {}'.format(sorted(state_entities)))
        cmds = self.langmodel.generate_all(state_entities, templates)
        if 'cookbook' in description and 'examine cookbook' not in cmds:
            cmds.append('examine cookbook')
        return cmds
