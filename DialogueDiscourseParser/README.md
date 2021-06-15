# Dialogue Discourse Parser For Meeting Summarization

This code is borrowed from [shizhouxing/DialogueDiscourseParsing](https://github.com/shizhouxing/DialogueDiscourseParsing).
We first train the **Deep Sequential Model** and then use the model to parse our own meetings.

## Requirements
```
* Python                 2.7
* tensorflow             1.3.0
* tensorflow-gpu         1.3.0
* tensorflow-tensorboard 0.1.8
```

## Data Format
Firstly, you should convert each meeting into the required data format. Note that relations are used for metric calculation, which is not useful for discourse parsing.
```
[
  {
    "edus": [
      {
        "text": "this is the kick-off meeting for our project .",
        "speaker": "B"
      },
      {
        "text": "and this is just what we're gonna be doing over the next twenty five minutes .",
        "speaker": "A"
      },
      ...
    ],
    "id": "xxxxx",
    "relations": [
      {
        "y": 1,
        "x": 0,
        "type": "Explanation"
      }
    ]
  }
]
```

## Pre-trained Model
You can download the pre-trained model [here](https://drive.google.com/drive/folders/1sbyXGL_hL8G6plFUd2fnCwx23yXE0mws?usp=sharing). Then put it under project dir, **DialogueDiscourseParser/model/\***

## Train from scratch
You can follow the instructions from [shizhouxing/DialogueDiscourseParsing](https://github.com/shizhouxing/DialogueDiscourseParsing) to train the parser.
Simply run `python main.py --is_train`, **we only modify the vocab_size to 2,500**.

## Parsing
* Choose dataset, **AMI** or **ICSI**.
    * In `run_test.py`, change **file_list**.
    * In `main.py`, change **data** to AMI or ICSI.
        * `tf.flags.DEFINE_string("data", "AMI", "data")`
    * In `main.py`, change **is_train** to False.
        * `tf.flags.DEFINE_boolean("is_train", False, "train model")`
    * In `utils.py`, in function `preview_data`, change output dir, and you should first create it.
        * `wf = codecs.open("./icsi_res/"+test_data.split(".")[0]+".txt","w","utf-8")`
    * In data dir, there should be two files **train.json, test.json (useless)**
* Start parsing.
    * Run `python run_test.py`
* You can get results from target dir.

## For your own data
1. You can first copy `train.json` and `test.json` under the AMI directory to your own data directory without any modification, results in `yourdata/train.json` and `yourdata/test.json` since these two files are only for code running rather than discourse parsing.
1. Then, for your meetings, each meeting is converted into a JSON file, namely, `meetingX.json` under the `yourdata` directory.
1. Finally, you may have a data directory `yourdata` including `meeting1.json`, `meeting2.json`, ..., `train.json`, and `test.json`.
1. Please note that `train.json`, and `test.json` are directly copied from AMI directory.


