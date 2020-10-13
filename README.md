# Source Code for VHDA #

This is the source code for Variational Hierarchical Dialog Autoencoder (VHDA), 
a hierarchical and recurrent VAE model for generating task-oriented dialog corpora ([arXiv](https://arxiv.org/abs/2001.08604)).


## Preparation ##

A fresh python environment (preferrably using Anaconda3) is recommended. This is the configuration of our original research environment.

  * OS: Ubuntu 18.04
  * CPU: i9-9900kf
  * RAM: 64GB
  * GPU: NVIDIA GTX 2080Ti / CUDA 10.0 / CuDNN 7.X
  * Python 3.7.3


### Git Clone ###

Clone this repository "recursively".

    git clone https://github.com/kaniblu/vhda --recursive

This repository contains FastText as a submodule.


### Install Python Packages ###

Install python packages from `requirements.txt`.

    pip install -r requirements.txt

### Install Tokenizer (Choose One) ###

Either install CoreNLP through the following instructions or install
`spacy` and download the basic English model.

#### Choice 1: Install Docker and CoreNLP ####

Docker is required to run the CoreNLP image. Assuming that the Docker daemon is running, run the following command to start a CoreNLP container instance.

    docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server

This is needed to tokenize utterances using CoreNLP.

#### Choice 2: Install SpaCy ####

After instaling the SpaCy library from `requirements.txt`, run 
`python -m spacy download en` to download the necessary nlp engine.

### Prepare Datasets ###

Run `python -m datasets.prepare` to prepare (preprocessing and format conversion) specific datasets.

If you installed CoreNLP is the previous step, run with `--tokenizer corenlp` option.

Requires an internet connection to download all raw data.

Preprocessed datasets will be stored under `data/`.

### Prepare Word Embeddings / FastText ###

Due to the size of word embeddings, we did not include the word embeddings required for our experiments.

Download the latest FastText english model from [the official website](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz).
Make sure that the path to the FastText model is updated in the yaml configurations under `dst/internal/configs/` (these are configurations for dialog state trackers).

For other embeddings (GloVe and character embeddings), a python package will take care of downloading them.

Also, make sure that the fasttext binaries under `third_party/fasttext` are updated.
This code was packaged with compiled binaries, but those might not work in a new environment.
Run `make clean && make` to re-compile.

## Running ##

These are the main entry points:

* `train`
* `train_eval`
* `train_eval_gda`
* `gda`
* `gda_helper`
* `generate`
* `interpolate`
* `interpolate_helper`
* `evaluate`
* `evaluate_helper`
* `datasets.prepare`
* `dst.internal.run`
* `dst.internal.evaluate`
* `dst.internal.multirun`

We will list all the basic commands here required for replication, but some options might have to be tweaked depending on the environment.

### Training Generator ###

To train a toy VHDA, run the following. This trains a small VHDA generator using a toy dataset and saves the results under `out/`. Use this to check whether everything is in place.

    python -m train [--gpu 0]

To replicate the full VHDA as described in the paper, run the following.

    for dataset in woz
    do 
        python -m train_eval \
            --save-dir out/$dataset \
            --model-path configs/vhda-dhier.yml \
            --data-dir data/$dataset/json/preprocessed \
            --epochs 10000 --batch-size 32 \
            --l2norm 0.000001 --validate-every 5 \
            --valid-batch-size 32 --gradient-clip 2.0 \
            --early-stop --early-stop-criterion "~nll" \
            --early-stop-patience 30 \
            --kl-mode "kl-mi" \
            --kld-schedule "[(0,0.0),(250000,1.0)]"
    done

This trains the model for the WoZ2.0 dataset and stores the results under `out/woz`.
Change `woz` to other datasets if needed.


### Training State Tracker ###

This code comes with a blazing fast implementation of the state tracker.

Run the following to verify whether DST can be run.

    python -m dst.internal.run

To run a baseline on WoZ2.0 without data augmentation using GCE+, run the following.

    for dataset in woz
    do 
        python -m dst.internal.multirun \
            --data-dir data/$dataset/json/preprocessed \
            --model-path dst/internal/configs/gce-ft-wed0.2.yml \
            --epochs 200 --batch-size 100 --gradient-clip 2.0 \
            --early-stop --runs 10
    done

This produces multiple runs of the baseline and the final results are aggregated under `out/summary.json`.

This assumes that FastText is properly installed. If not, refer to the preparation guide.

### Training Tracker with GDA ###

To train trackers with GDA enabled, use `gda_helper`:

    for dataset in woz
    do 
        python -m gda_helper \
            --dataset $dataset \
            --exp-dir <path-to-gen> \
            --dst-model gce-ft-wed0.2 \
            --scale 1.0 1.0 1.0 1.0 1.0 \
            --gen-runs 3 \
            --dst-runs 3 \
            --epochs 200 \
            --save-dir <save-dir> \
            --multiplier 1.0 \
            --batch-size 100 
    done

Specify the directory to the trained generator using the `--exp-dir` option and the directory to save the results in the `--save-dir` option.

Above command will generate samples from a pretrained generator (VHDA) and train a new set of trackers using the augmented dataset using `gce-ft-wed0.2` configuration of the tracker model.

The aggregated results of the DST runs will be stored under `<save-dir>/dst-summary.json`.

## Notes ##

* The training time for VHDA is about 20 minutes for WoZ2.0 and the training time for GCE+ is about 10 minutes for WoZ2.0.
* Available datasets can be checked under `data/` (after running data preparation).
    * Make soft links to the MultiWoZ-Hotel and MultiWoZ-Rest datasets if needed (these datasets are created under `data/multiwoz/domains/` directory)
* Available generator models are listed under `configs/` and available tracker models are listed under `dst/internal/configs/`.
* Generation samples can be checked under the results of `gda_helper`.

## Citation ##

Please cite the main paper as follows.

    @article{yoo2020variational,
        title={Variational Hierarchical Dialog Autoencoder for Dialogue State Tracking Data Augmentation},
        author={Yoo, Kang Min and Lee, Hanbit and Dernoncourt, Franck and Bui, Trung and Chang, Walter and Lee, Sang-goo},
        journal={arXiv preprint arXiv:2001.08604},
        year={2020}
    }