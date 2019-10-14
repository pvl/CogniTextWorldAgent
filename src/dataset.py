import os
import re
import random
import shutil
import glob
import argparse
from tqdm import tqdm
import gym
from textworld import EnvInfos
import textworld
import textworld.gym
import pandas as pd
from commandgenerator import CommandModel
from nerdataset import get_category

random.seed(42)

DEBUG = False


def get_cv_games(datapath, block='valid'):
    games = [game for game in os.listdir(datapath)
             if game.endswith('.ulx')]
    with open('valid200.txt') as f:
        valid = [g.strip() for g in f.readlines()]
    if block == 'valid':
        games = [os.path.join(datapath, game)
                 for game in games if game in valid]
        return games
    else:
        games = [os.path.join(datapath, game)
                 for game in games if game not in valid]
        return games


def train_test_split(datapath):
    """ split the data into train/validation """
    filenames = os.listdir(os.path.join(datapath, 'train'))
    basename = set()
    for fn in filenames:
        parts = os.path.splitext(fn)
        if len(parts) == 2:
            base, ext = parts
            if ext in ('.z8', '.ulx', '.json'):
                basename.add(base)
            else:
                print('ignored ', fn)
        else:
            print('Failed for ', fn)

    basename = list(basename)
    random.shuffle(basename)
    test_names = basename[:400]
    os.makedirs(os.path.join(datapath, 'valid/'), exist_ok=True)
    for name in test_names:
        fns = glob.glob(os.path.join(datapath, 'train', f'{name}.*'))
        for fn in fns:
            shutil.move(fn, os.path.join(datapath, 'valid/'))


def get_game_name(gamepath):
    return os.path.split(gamepath)[-1]


class Walkthrough:
    def __init__(self, walkthrough, include_cookbook=True):
        if include_cookbook:
            self.walkthrough = self.filter_walkthrough(walkthrough)
        else:
            self.walkthrough = walkthrough
        self.pos = 0
        self.include_cookbook = include_cookbook
        self.doors = ['plain door', 'sliding patio door', 'front door',
                      'commercial glass door', 'sliding door', 'patio door',
                      'fiberglass door', 'barn door', 'screen door',
                      'wooden door', 'frosted glass door',
                      'frosted-glass door']

    def filter_walkthrough(self, walkthrough):
        # remove open door
        walkthrough = [cmd for cmd in walkthrough
                       if not (cmd.startswith('open')
                               and cmd.endswith('door'))]
        return walkthrough

    def get_next_command(self, description, admissible_commands,
                         seen_cookbook):
        if self.include_cookbook:
            return self.get_next_command_cookbook(description,
                                                  admissible_commands,
                                                  seen_cookbook)
        else:
            return self.get_next_command_basic(admissible_commands,
                                               seen_cookbook)

    def get_next_command_basic(self, admissible_commands, seen_cookbook):
        if self.pos < len(self.walkthrough):
            cmd = self.walkthrough[self.pos]
            self.pos += 1
            return cmd
        else:
            return None

    def get_first_closed_door(self, description):
        for door in self.doors:
            check = f'closed {door}'
            if check in description:
                return door
        return ''

    def get_next_command_cookbook(self, description, admissible_commands,
                                  seen_cookbook):
        if not seen_cookbook and 'examine cookbook' in admissible_commands:
            return 'examine cookbook'
        elif self.get_first_closed_door(description):
            ent = self.get_first_closed_door(description)
            return f'open {ent}'
        elif self.pos < len(self.walkthrough):
            cmd = self.walkthrough[self.pos]
            self.pos += 1
            return cmd
        else:
            return None


def run_game(gamefile, walkthrough_ext=None):
    """ runs a game following the walkthrough and extracts the information """
    if DEBUG:
        print('Game name:', gamefile)
    requested_infos = EnvInfos(description=True, inventory=True,
                               admissible_commands=True, objective=True,
                               verbs=True, command_templates=True,
                               entities=True, max_score=True, has_won=True,
                               has_lost=True, extras=["recipe", "walkthrough"])
    env_id = textworld.gym.register_games([gamefile], requested_infos)
    env = gym.make(env_id)
    obs, infos = env.reset()
    if walkthrough_ext:
        walk = Walkthrough(walkthrough_ext, include_cookbook=False)
    else:
        walk = Walkthrough(infos['extra.walkthrough'])
    if DEBUG:
        print('Walk:', walk.walkthrough)
    seen_cookbook = False
    gamename = get_game_name(gamefile)
    gamesteps = []
    idx = 0
    while True:
        cmd = walk.get_next_command(infos['description'],
                                    infos['admissible_commands'],
                                    seen_cookbook)
        if cmd is None:
            break
        if DEBUG:
            print('cmd:', cmd)
        step = {
            'gamename': gamename,
            'step': idx,
            'description': infos['description'],
            'inventory': infos['inventory'],
            'recipe': infos['extra.recipe'],
            'admissible_commands': infos['admissible_commands'],
            'entities': infos['entities'],
            'verbs': infos['verbs'],
            'command_templates': infos['command_templates'],
            'objective': infos['objective'],
            'max_score': infos['max_score'],
            'has_won': infos['has_won'],
            'has_lost': infos['has_lost'],
            'walkthrough': infos['extra.walkthrough'],
            'seen_cookbook': seen_cookbook,
            'command': cmd
        }
        obs, scores, dones, infos = env.step(cmd)
        step['reward'] = scores
        step['obs'] = obs
        if DEBUG:
            print('obs:', obs)
        gamesteps.append(step)
        if cmd == 'examine cookbook':
            seen_cookbook = True
        idx += 1

    assert dones
    return gamesteps


class ReworkWalkthrough:
    def __init__(self, walkthrough):
        self.revmap = {'north': 'south', 'west': 'east',
                       'south': 'north', 'east': 'west'}
        self.walkthrough = self.rework(walkthrough)

    def reverse_cmd(self, cmd):
        if cmd.startswith('go'):
            direction = cmd.split()[-1]
            return 'go {}'.format(self.revmap[direction])
        return cmd

    def rework(self, walkthrough):
        nwalk = []
        droplist = []
        take_navigate = []
        seen_cookbook = False
        record_take = False
        for w in walkthrough:
            if w == 'examine cookbook':
                seen_cookbook = True
                nwalk.append(w)
                # now need to go back to take items
                for cmd in reversed(take_navigate):
                    if not cmd.startswith('take'):
                        nwalk.append(self.reverse_cmd(cmd))
                    else:
                        # need to empty droplist here
                        for dropcmd in droplist:
                            nwalk.append(dropcmd)
                        droplist = []
                        nwalk.append(cmd)
                # now we need to return to cookbook location
                for cmd in take_navigate:
                    if not cmd.startswith('take'):
                        nwalk.append(cmd)
                record_take, take_navigate = False, []
            elif w.startswith('take') and not seen_cookbook:
                record_take = True
                take_navigate.append(w)
            elif w.startswith('take') and seen_cookbook:
                for cmd in droplist:
                    nwalk.append(cmd)
                droplist = []
                nwalk.append(w)
            elif w.startswith('drop') and not seen_cookbook:
                droplist.append(w)
            else:
                if record_take:
                    take_navigate.append(w)
                nwalk.append(w)
        return nwalk


def extract_walkthrough_dataset(games, walkthrough_ext_dict=None):
    """
    runs a sequence of games and collects all information into a dataframe
    """
    gamesteps = []
    for game in tqdm(games):
        if walkthrough_ext_dict:
            walkext = walkthrough_ext_dict[get_game_name(game)]
        else:
            walkext = None
        gamesteps += run_game(game, walkext)
    return pd.DataFrame(gamesteps)


def extract_datasets(datapath, outputpath, use_walkthrough_ext=False):
    """ runs all games and saves dataframes with state/command """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    if use_walkthrough_ext:
        gtr = pd.read_csv(os.path.join(outputpath, 'enhanced_walkthrough.csv'))
        gtr['walk_ext'] = gtr.walk_ext.apply(eval)
        walkext_dict = dict(zip(gtr.gamename.values, gtr.walk_ext.values))
        sufix = '_cookbook'
    else:
        walkext_dict, sufix = None, ''

    valid_games = get_cv_games(datapath, 'valid')
    valid_data = extract_walkthrough_dataset(valid_games, walkext_dict)
    valid_data.to_csv(os.path.join(outputpath,
                      f'walkthrough_valid{sufix}.csv'), index=False)
    train_games = get_cv_games(datapath, 'train')
    train_data = extract_walkthrough_dataset(train_games, walkext_dict)
    train_data.to_csv(os.path.join(outputpath,
                      f'walkthrough_train{sufix}.csv'), index=False)


def enhanced_walkthrough(basepath):
    """
    based on the extracted walkthrough make an enhanced walkthrough
    that first finds the cookbook and then does the take/drop actions
    """
    valid = pd.read_csv(os.path.join(basepath, 'walkthrough_valid.csv'))
    train = pd.read_csv(os.path.join(basepath, 'walkthrough_train.csv'))
    gtr = train.groupby('gamename').command.apply(list).to_frame(
                        'walk_cookbook').reset_index()
    gva = valid.groupby('gamename').command.apply(list).to_frame(
                        'walk_cookbook').reset_index()
    gtr = pd.concat([gtr, gva])

    def make_walk_ext(walk):
        wt = ReworkWalkthrough(walk)
        return wt.walkthrough

    gtr['walk_ext'] = gtr.walk_cookbook.apply(make_walk_ext)
    gtr.to_csv(os.path.join(basepath, 'enhanced_walkthrough.csv'), index=False)


def postprocess_recipe(wtr):
    def mark_cookbook(s):
        res = []
        seen = False
        for cmd in s.values:
            if seen:
                res.append(1)
            elif cmd == 'examine cookbook':
                res.append(0)
                seen = True
            else:
                res.append(0)
        return res

    wtr['seen_cookbook'] = wtr.groupby('gamename').command\
                              .transform(mark_cookbook)
    wtr.loc[wtr.seen_cookbook == 0, 'recipe'] = 'missing recipe'
    wtr = wtr.drop('seen_cookbook', axis=1)
    return wtr


def fix_recipe(basepath):
    train = pd.read_csv(os.path.join(basepath,
                        'walkthrough_train_cookbook.csv'))
    valid = pd.read_csv(os.path.join(basepath,
                        'walkthrough_valid_cookbook.csv'))
    train = postprocess_recipe(train)
    valid = postprocess_recipe(valid)
    train.to_csv(os.path.join(basepath, 'walkthrough_train_cookbook.csv'),
                 index=False)
    valid.to_csv(os.path.join(basepath, 'walkthrough_valid_cookbook.csv'),
                 index=False)


def simplify_command(cmd):
    nc = cmd
    mobj = re.search(r'\b((in|with|on|into|from) [\w\- ]+)', cmd)
    if mobj:
        part = mobj.group(1)
        if part not in ('with oven', 'with stove', 'with BBQ'):
            nc = cmd.replace(part, '').strip()
    return nc


def entity_category_fix(entity):
    if entity == 'frosted-glass door':
        return 'D'
    return get_category(entity)


def generate_datasets_commands(basepath):
    """ generate a dataset with commands and entities extracted"""

    train = pd.read_csv(os.path.join(basepath,
                        'walkthrough_train_cookbook.csv'))
    valid = pd.read_csv(os.path.join(basepath,
                        'walkthrough_valid_cookbook.csv'))
    for c in ['entities', 'command_templates']:
        train[c] = train[c].apply(eval)
        valid[c] = valid[c].apply(eval)

    def _extract_commands(s):
        cm = CommandModel()
        entity_cat = [(e, entity_category_fix(e)) for e in s.entities]
        return cm.generate_all(entity_cat, s.command_templates)

    valid['gen_commands'] = valid.apply(_extract_commands, axis=1)
    train['gen_commands'] = train.apply(_extract_commands, axis=1)

    # the correct command also needs to be simplified
    valid['command'] = valid['command'].apply(simplify_command)
    train['command'] = train['command'].apply(simplify_command)

    valid.to_csv(os.path.join(basepath, 'walkthrough_valid_commands_real.csv'),
                 index=False)
    train.to_csv(os.path.join(basepath, 'walkthrough_train_commands_real.csv'),
                 index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Generate datasets for QA model training")
    parser.add_argument('datapath', type=str, help="path to the folder games")
    parser.add_argument('--output', type=str, default='../datafiles',
                        help="path to save the processed datasets")
    args = parser.parse_args()
    extract_datasets(args.datapath, args.output)
    enhanced_walkthrough(args.output)
    extract_datasets(args.datapath, args.output, use_walkthrough_ext=True)
    fix_recipe(args.output)
    generate_datasets_commands(args.output)
