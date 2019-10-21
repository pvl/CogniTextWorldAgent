# First TextWorld Challenge - First Place Solution

This is the winning solution of the Microsoft Research [First TextWorld Problems: A Reinforcement and Language Learning Challenge](https://www.microsoft.com/en-us/research/project/textworld/).

A detailed article with the challenge and solution was published in this
[Microsoft blog post](https://www.microsoft.com/en-us/research/blog/first-textworld-problems-the-competition-using-text-based-games-to-advance-capabilities-of-ai-agents/).

A [second blog post](https://medium.com/@pvl/first-textworld-challenge-first-place-solution-notes-d081bb9dee11) with more details on the agent design and implementation.

TextWorld is text games simulator environment for NLP research that in my opinion makes research extremely fun.

## Installation

This solution requires Python 3.6+. It was tested on a Linux system with a CUDA GPU.

It is recommended to install the required packages in a virtual environment.

```
python3 -m venv twenv
source twenv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader 'punkt'
```

The [Apex package](https://github.com/NVIDIA/apex) is also required to train the model in the GPU with 16-bit to allow larger batches in the limited GPU memory (this probably can be optional when running on a
GPU with 16Gb or more).


## Model training

To train the models run the following command with the folder having the games (games need to be downloaded from the TextWorld website).

```
python src/runtrain.py <games folder>
```

After training the models are saved in `qamodel` and `nermodel` in the `outputs` folder.

## Test the agent

In the competition the agent executed in an environment without Apex. To test create a new env without Apex and run the following command

```
python src/playgame.py --display <games file>
```

This will output to the screen the game with the agent commands. You can also execute the command with a folder having several games.

## NER model

For named entities this solution uses the [BERT-NER model](https://github.com/kamalkraj/BERT-NER).
