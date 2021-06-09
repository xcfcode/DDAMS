# DDAMS

This is the **pytorch** code for our **IJCAI 2021** paper **Dialogue Discourse-Aware Graph Model and Data Augmentation for Meeting Summarization** [[Arxiv Preprint]](https://arxiv.org/abs/2012.03502).

<p align="center">
  <img src="pic/1.png" width="400">
</p>

## Update

* **2021.6.9** update pretrained models for AMI and ICSI. [here](https://drive.google.com/drive/folders/1m7RxAU5GxPxri1i9fTZbDH-H4AMb2xp_?usp=sharing), under the `qg_pretrain` dir;
* **2021.6.5** update [Dialogue Discourse Parser](https://github.com/xcfcode/DDAMS/tree/main/DialogueDiscourseParser);

## Outputs
Output summaries are available at [AMI](https://github.com/xcfcode/DDAMS/blob/main/summaries/ami_summary.txt) and [ICSI](https://github.com/xcfcode/DDAMS/blob/main/summaries/icsi_summary.txt).

## Requirements
* We use Conda python 3.7 and strongly recommend that you create a new environment: `conda create -n ddams python=3.7`.
* Run the following command: `pip install -r requirements.txt`. 
    * We use **[pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)** for GNN implementation.

## Data
You can download data [here](https://drive.google.com/drive/folders/1knrWtO8V8D8s8tqztRENumEx19t-CCgp?usp=sharing), put the data under the project dir **DDAMS/data/xxx**.

* **data/ami**
    * **data/ami/ami**: preprocessed meeting data
    * **data/ami/ami_qg**: pseudo summarization data.
    * **data/ami/ami_reference**: golden reference for test file.
* **data/icsi**
    * **data/icsi/icsi**: preprocessed meeting data
    * **data/icsi/icsi_qg**: pseudo summarization data.
    * **data/icsi/icsi_reference**: golden reference for test file.
* **data/glove**: pre-trained word embedding `glove.6B.300d.txt`.

## Reproduce Results
You can follow the following steps to reproduce the best results in our paper.

### download checkpoints
Download checkpoints [here](https://drive.google.com/drive/folders/1m7RxAU5GxPxri1i9fTZbDH-H4AMb2xp_?usp=sharing). Put the checkpoints, including `AMI.pt` and `ICSI.pt`, under the project dir **DDAMS/models/xx.pt**.

### translate
Produce final summaries.

For AMI, we can get **summaries/ami_summary.txt**.

```
CUDA_VISIBLE_DEVICES=X python translate.py -batch_size 1 \
               -src data/ami/ami/test.src \
               -tgt data/ami/ami/test.tgt \
               -seg data/ami/ami/test.seg \
               -speaker data/ami/ami/test.speaker \
               -relation data/ami/ami/test.relation \
               -beam_size 10 \
               -share_vocab \
               -dynamic_dict \
               -replace_unk \
               -model models/AMI.pt \
               -output summaries/ami_summary.txt \
               -block_ngram_repeat 3 \
               -gpu 0 \
               -min_length 280 \
               -max_length 450
```

For ICSI, we can get **summaries/icsi_summary.txt**.

```
CUDA_VISIBLE_DEVICES=x python translate.py -batch_size 1 \
               -src data/icsi/icsi/test.src \
               -seg data/icsi/icsi/test.seg \
               -speaker data/icsi/icsi/test.speaker \
               -relation data/icsi/icsi/test.relation \
               -beam_size 10 \
               -share_vocab \
               -dynamic_dict \
               -replace_unk \
               -model models/ICSI.pt \
               -output summaries/icsi_summary.txt \
               -block_ngram_repeat 3 \
               -gpu 0 \
               -min_length 400 \
               -max_length 550
```


### remove tags
`<t>` and `</t>` will raise errors for ROUGE test. So we should first remove them. (following [OpenNMT](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/Summarization.md))

```
sed -i 's/ <\/t>//g' summaries/ami_summary.txt
sed -i 's/<t> //g' summaries/ami_summary.txt
sed -i 's/ <\/t>//g' summaries/icsi_summary.txt
sed -i 's/<t> //g' summaries/icsi_summary.txt
```

### test rouge score
* Change `pyrouge.Rouge155()` to your local path.

Output format `>> ROUGE(1/2/L): xx.xx-xx.xx-xx.xx`

```
python test_rouge.py -c summaries/ami_summary.txt
python test_rouge_icsi.py -c summaries/icsi_summary.txt
```

### ROUGE score
You will get following ROUGE scores.

||ROUGE-1| ROUGE-2 | ROUGE-L |
| :---: | :---: | :---: | :---: |
| AMI | 53.15 | 22.32 | 25.67|
| ICSI | 40.41 | 11.02 | 19.18 |


## From Scratch

### Dialogue Discourse Parser
Please refer to **DialogueDiscourseParser** dir.

### For AMI
#### Preprocess
(1) Preprocess AMI dataset.

```
python preprocess.py -train_src data/ami/ami/train.src \
                     -train_tgt data/ami/ami/train.tgt \
                     -train_seg data/ami/ami/train.seg \
                     -train_speaker data/ami/ami/train.speaker \
                     -train_relation data/ami/ami/train.relation \
                     -valid_src data/ami/ami/valid.src \
                     -valid_tgt data/ami/ami/valid.tgt \
                     -valid_seg data/ami/ami/valid.seg \
                     -valid_speaker data/ami/ami/valid.speaker \
                     -valid_relation data/ami/ami/valid.relation \
                     -save_data data/ami/AMI \
                     -dynamic_dict \
                     -share_vocab \
                     -lower \
                     -overwrite
```

(2) Create pre-trained word embeddings.

```
python embeddings_to_torch.py -emb_file_both data/glove/glove.6B.300d.txt \
-dict_file data/ami/AMI.vocab.pt \
-output_file data/ami/ami_embeddings
```

(3) Preprocess pseudo summarization dataset.

```
python preprocess.py -train_src data/ami/ami_qg/train.src \
                     -train_tgt data/ami/ami_qg/train.tgt \
                     -train_seg data/ami/ami_qg/train.seg \
                     -train_speaker data/ami/ami_qg/train.speaker \
                     -train_relation data/ami/ami_qg/train.relation \
                     -save_data data/ami/AMIQG \
                     -lower \
                     -overwrite \
                     -shard_size 500 \
                     -dynamic_dict \
                     -share_vocab
```

#### Train
(1) we first pre-train our DDAMS on the pseudo summarization dataset.

- run the following command to save config file (`-save_config`). 
- remove `-save_config` and rerun the command to start the training process.

```
CUDA_VISIBLE_DEVICES=X python train.py -save_model ami_qg_pretrain/AMI_qg\
           -data data/ami/AMIQG \
           -speaker_type ami \
           -batch_size 64 \
           -learning_rate 0.001 \
           -share_embeddings \
           -share_decoder_embeddings \
           -copy_attn \
           -reuse_copy_attn \
           -report_every 30 \
           -encoder_type hier3 \
           -global_attention general \
           -save_checkpoint_steps 500 \
           -start_decay_steps 1500 \
           -pre_word_vecs_enc data/ami/ami_embeddings.enc.pt \
           -pre_word_vecs_dec data/ami/ami_embeddings.dec.pt \
           -log_file logs/ami_qg_pretrain.txt \
           -save_config logs/ami_qg_pretrain.txt
```

(2) fine-tuning on AMI.

```
CUDA_VISIBLE_DEVICES=X python train.py -save_model ami_final/AMI \
           -data data/ami/AMI \
           -speaker_type ami \
           -train_from ami_qg_pretrain/xxx.pt  \
           -reset_optim all \
           -batch_size 1 \
           -learning_rate 0.0005 \
           -share_embeddings \
           -share_decoder_embeddings \
           -copy_attn \
           -reuse_copy_attn \
           -encoder_type hier3 \
           -global_attention general \
           -dropout 0.5 \
           -attention_dropout 0.5 \
           -start_decay_steps 500 \
           -decay_steps 500 \
           -log_file logs/ami_final.txt \
           -save_config logs/ami_final.txt
```

#### Translate

```
CUDA_VISIBLE_DEVICES=X python translate.py -batch_size 1 \
               -src data/ami/ami/test.src \
               -tgt data/ami/ami/test.tgt \
               -seg data/ami/ami/test.seg \
               -speaker data/ami/ami/test.speaker \
               -relation data/ami/ami/test.relation \
               -beam_size 10 \
               -share_vocab \
               -dynamic_dict \
               -replace_unk \
               -model xxx.pt \
               -output xxx.txt \
               -block_ngram_repeat 3 \
               -gpu 0 \
               -min_length 280 \
               -max_length 450
```

### For ICSI

#### Preprocess
(1) Preprocess ICSI dataset.

```
python preprocess.py -train_src data/icsi/icsi/train.src \
                     -train_tgt data/icsi/icsi/train.tgt \
                     -train_seg data/icsi/icsi/train.seg \
                     -train_speaker data/icsi/icsi/train.speaker \
                     -train_relation data/icsi/icsi/train.relation \
                     -valid_src data/icsi/icsi/valid.src \
                     -valid_tgt data/icsi/icsi/valid.tgt \
                     -valid_seg data/icsi/icsi/valid.seg \
                     -valid_speaker data/icsi/icsi/valid.speaker \
                     -valid_relation data/icsi/icsi/valid.relation \
                     -save_data data/icsi/ICSI \
                     -src_seq_length 20000 \
                     -src_seq_length_trunc 20000 \
                     -tgt_seq_length 700 \
                     -tgt_seq_length_trunc 700 \
                     -dynamic_dict \
                     -share_vocab \
                     -lower \
                     -overwrite
```

(2) Create pre-trained word embeddings.

```
python embeddings_to_torch.py -emb_file_both data/glove/glove.6B.300d.txt \
-dict_file data/icsi/ICSI.vocab.pt \
-output_file data/icsi/icsi_embeddings
```

(3) Preprocess pseudo summarization dataset.

```
python preprocess.py -train_src data/icsi/icsi_qg/train.src \
                     -train_tgt data/icsi/icsi_qg/train.tgt \
                     -train_seg data/icsi/icsi_qg/train.seg \
                     -train_speaker data/icsi/icsi_qg/train.speaker \
                     -train_relation data/icsi/icsi_qg/train.relation \
                     -save_data data/icsi/ICSIQG \
                     -lower \
                     -overwrite \
                     -shard_size 500 \
                     -dynamic_dict \
                     -share_vocab
```

#### Train

(1) pre-training. 

```
CUDA_VISIBLE_DEVICES=X python train.py -save_model icsi_qg_pretrain/ICSI \
           -data data/icsi/ICSIQG \
           -speaker_type icsi \
           -batch_size 64 \
           -learning_rate 0.001 \
           -share_embeddings \
           -share_decoder_embeddings \
           -copy_attn \
           -reuse_copy_attn \
           -report_every 30 \
           -encoder_type hier3 \
           -global_attention general \
           -save_checkpoint_steps 500 \
           -start_decay_steps 1500 \
           -pre_word_vecs_enc data/icsi/icsi_embeddings.enc.pt \
           -pre_word_vecs_dec data/icsi/icsi_embeddings.dec.pt \
           -log_file logs/icsi_qg_pretrain.txt \
           -save_config logs/icsi_qg_pretrain.txt
```

(2) fine-tuning on ICSI.

```
CUDA_VISIBLE_DEVICES=X python train.py -save_model icsi_final/ICSI \
           -data data/icsi/ICSI \
           -speaker_type icsi \
           -train_from icsi_qg_pretrain/xxx.pt  \
           -reset_optim all \
           -batch_size 1 \
           -learning_rate 0.0005 \
           -share_embeddings \
           -share_decoder_embeddings \
           -copy_attn \
           -reuse_copy_attn \
           -encoder_type hier3 \
           -global_attention general \
           -dropout 0.5 \
           -attention_dropout 0.5 \
           -start_decay_steps 1000 \
           -decay_steps 100 \
           -save_checkpoint_steps 50 \
           -valid_steps 50 \
           -log_file logs/icsi_final.txt \
           -save_config logs/icsi_final.txt
```

#### Translate

```
CUDA_VISIBLE_DEVICES=x python translate.py -batch_size 1 \
               -src data/icsi/icsi/test.src \
               -seg data/icsi/icsi/test.seg \
               -speaker data/icsi/icsi/test.speaker \
               -relation data/icsi/icsi/test.relation \
               -beam_size 10 \
               -share_vocab \
               -dynamic_dict \
               -replace_unk \
               -model xxx.pt \
               -output xxx.txt \
               -block_ngram_repeat 3 \
               -gpu 0 \
               -min_length 400 \
               -max_length 550
```

## Test Rouge
(1) Before ROUGE test, we should first remove special tags: <t> </t>.

```
sed -i 's/ <\/t>//g' xxx.txt
sed -i 's/<t> //g' xxx.txt
```

(2) Test rouge

```
python test_rouge.py -c summaries/xxx.txt
python test_rouge_icsi.py -c summaries/xxx.txt
```



