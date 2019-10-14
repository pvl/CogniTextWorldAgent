#!/usr/bin/env python3

# Modified version of the TextWorld startkit evaluation code

import argparse
import glob
import json
import multiprocessing
import os
import requests
import subprocess
import sys
import tempfile
import gym
import textworld
import textworld.gym
import time
import tqdm
from termcolor import colored


NB_EPISODES = 10
MAX_EPISODE_STEPS = 100
TIMEOUT = 12 * 30 * 60  # 12 hours
DISPLAY_GAME = False

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = textworld.EnvInfos(
    max_score=True, has_won=True, has_lost=True,                    # Handicap 0
    description=True, inventory=True, objective=True,               # Handicap 1
    verbs=True, command_templates=True,                             # Handicap 2
    entities=True,                                                  # Handicap 3
    extras=["recipe"],                                              # Handicap 4
    admissible_commands=True,                                       # Handicap 5
)


def display(text, type=''):
    def formatproba(data, color='blue'):
        data = list(reversed(sorted(data, key=lambda x: x[1])))
        data = data[:5]
        lines = []
        for k,v in data:
            lines.append('  {:4.2f} {}'.format(v,k))
        if lines:
            maxsize = max([len(k) for k in lines])
            print(colored('-'*(maxsize+2), color))
            for line in lines:
                print(colored(line, color))
            print(colored('-'*(maxsize+2), color))

    if DISPLAY_GAME:
        if type == 'cmd':
            print(colored(text, 'green'))
        elif type == 'title':
            print(colored(text, 'red'))
        elif type == 'cmdproba':
            formatproba(text)
        else:
            print(text)

            
def _validate_requested_infos(infos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


class _ReplayAgent:
    """
    An agent that replays the actions of another agent.
    """

    def __init__(self, stats):
        self._stats = stats
        self._game = None
        self._episode = 0
        self._step = 0

    def train(self):
        pass

    def eval(self):
        pass

    def select_additional_infos(self):
        infos = textworld.EnvInfos()
        for info in self._stats["requested_infos"]:
            if info in AVAILABLE_INFORMATION.extras:
                infos.extras.append(info)
            else:
                setattr(infos, info, True)
        return infos

    def act(self, obs, scores, dones, infos):
        if all(dones):
            self._episode += 1
            self._step = 0
            return

        if infos["_name"] != self._game:
            self._game = infos["_name"]
            self._episode = 0

        step = self._step
        self._step += 1

        command = self._stats["games"][self._game]["runs"][self._episode]["commands"][step]
        return [command]


def _play_game(agent_class, agent_class_args, gamefile):
    game_name = os.path.basename(gamefile)
    display(game_name, type='title')
    if agent_class_args:
        agent = agent_class(agent_class_args)
    else:
        agent = agent_class()

    agent.eval()
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    # Turn on flags needed for the evaluation.
    requested_infos.has_won = True
    requested_infos.has_lost = True
    requested_infos.max_score = True

    stats = {}
    start_time = time.time()

    stats["runs"] = []

    name = "test_{}".format(hash(gamefile))
    env_id = textworld.gym.register_games([gamefile], requested_infos,
                                            max_episode_steps=MAX_EPISODE_STEPS,
                                            name=name)
    env_id = textworld.gym.make_batch(env_id, batch_size=1)
    env = gym.make(env_id)

    for no_episode in range(NB_EPISODES):
        obs, infos = env.reset()

        all_commands = []
        scores = [0] * len(obs)
        dones = [False] * len(obs)
        steps = [0] * len(obs)
        while not all(dones):
            # Increase step counts.
            steps = [step + int(not done) for step, done in zip(steps, dones)]

            # HACK to get the replay agent the current game
            if isinstance(agent, _ReplayAgent):
                infos["_name"] = game_name

            commands, cmd_proba = agent.act_detail(obs, scores, dones, infos)
            display(obs[0])
            display(cmd_proba[0], type='cmdproba')
            display(commands[0], type='cmd')
            all_commands.append(commands)
            obs, scores, dones, infos = env.step(commands)

        # Let the agent knows the game is done.
        agent.act(obs, scores, dones, infos)
        display(obs[0])

        # Collect stats
        stats["runs"].append({})
        stats["runs"][no_episode]["score"] = scores[0]
        stats["runs"][no_episode]["steps"] = steps[0]
        stats["runs"][no_episode]["commands"] = [cmds[0] for cmds in all_commands]
        stats["runs"][no_episode]["has_won"] = infos["has_won"][0]
        stats["runs"][no_episode]["has_lost"] = infos["has_lost"][0]

    env.close()
    stats["max_scores"] = infos["max_score"][0]
    elapsed = time.time() - start_time
    stats["duration"] = elapsed
    stats["model_evaluations"] = agent.agent.n_model_evals

    return {game_name: stats}, requested_infos.basics + requested_infos.extras


def evaluate(agent_class, agent_class_args, game_files, nb_processes):
    stats = {"games": {}, "requested_infos": []}

    desc = "Evaluating {} games".format(len(game_files))

    def _assemble_results(args):
        data, requested_infos = args
        stats["games"].update(data)
        stats["requested_infos"] = requested_infos

        game_name, infos = list(data.items())[0]
        total_scores = sum(d["score"] for d in infos["runs"])
        total_steps = sum(d["steps"] for d in infos["runs"])

        desc = "{:2d} / {}:\t{}".format(total_scores, total_steps, game_name)

    if nb_processes > 1:
        pool = multiprocessing.Pool(nb_processes)
        for game_file in game_files:
            pool.apply_async(_play_game, (agent_class, agent_class_args, game_file), callback=_assemble_results)

        pool.close()
        pool.join()

    else:
        for game_file in game_files:
            data = _play_game(agent_class, agent_class_args, game_file)
            _assemble_results(data)

    return stats


def _run_evaluation(agent_class, args, agent_class_args=None):
    # multiple games in a folder
    if os.path.isdir(args.games_dir):
        games = glob.glob(os.path.join(args.games_dir, "**/*.ulx"), recursive=True)
    # single game
    elif args.games_dir.endswith('.ulx'):
        games = [args.games_dir]

    stats = evaluate(agent_class, agent_class_args, games, args.nb_processes)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        with open(args.output, "w") as f:
            json.dump(stats, f)



def main():
    global NB_EPISODES, DISPLAY_GAME
    
    parser = argparse.ArgumentParser(description="Evaluate an agent.")
    parser.add_argument("submission_dir")
    parser.add_argument("games_dir")
    parser.add_argument("--output", default="")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--nb-episodes", type=int, default=1)
    parser.add_argument("--nb-processes", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.nb_processes = 1
        
    DISPLAY_GAME = args.display
    NB_EPISODES = args.nb_episodes
    args.submission_dir = os.path.abspath(args.submission_dir)
    args.games_dir = os.path.abspath(args.games_dir)
    if args.output:
        args.output = os.path.abspath(args.output)
    os.chdir(args.submission_dir)
    sys.path = [args.submission_dir] + sys.path
    from custom_agent import CustomAgent
    _run_evaluation(CustomAgent, args, {'docker': False})


if __name__ == "__main__":
    main()
