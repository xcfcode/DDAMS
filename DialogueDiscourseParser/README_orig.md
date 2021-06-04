# Dialogue Discourse Parsing

Code for the AAAI-19 paper "A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues" (Shi and Huang, 2019)

## Paper

Our paper is now available on [arXiv](https://arxiv.org/abs/1812.00176), and please kindly cite it:

```
@inproceedings{shi2019deep,
  title={A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues},
  author={Shi, Zhouxing and Huang, Minlie},
  booktitle={AAAI},
  year={2019}
}
```

## Requirements

* Python 2.7
* Tensorflow 1.3

## Dataset

We use the [STAC corpus](https://www.irit.fr/STAC/corpus.html).

You may transform the original dataset into JSON files by:

```
python data_pre.py <input_dir> <output_json_file>
```

We also use 100-dimensional [Glove](https://nlp.stanford.edu/projects/glove/) word vectors.

## Run

```
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Available options can be found at the top of `main.py`.

For example, to train the model with default settings:

```
python main.py --is_train
```

