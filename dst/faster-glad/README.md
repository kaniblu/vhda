# Faster GLAD 

In terms of functionality, this fork is identical to the [base code](https://github.com/saleforce/glad),
except that the training speed has been greatly improved.

On a machine with the GTX 1080 GPU and the 6-th gen Intel CPU, the speed-up is about
7 times.

| Code       | Dataset         | Training Time       | Training Time per Epoch   |
| ---------- | --------------- | ------------------- | ------------------------- |
| [Original](https://github.com/salesforce/glad)   | WoZ   | 58.5 minutes      | 70 seconds
| This fork  | WoZ             | 7.83 minutes        | 9.5 seconds               |

The speedup is achieved by the following changes:

  - [sequence masks](https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1) 
    are used in place of python `for` loops for padding `-inf` before applying
    `torch.softmax`. (See self-attention and `attend` functions in `glad.py`.)
  - the dialogue act inference now operates on the tensor-level instead of looping
    through each data sample in a mini-batch. (See `Model.forward` method.)

Using default options, current implementation achieves 87.1% joint goal accuracy,
92.5% turn inform accuracy and 97.1% turn request accuracy.

Following is the original README file.

# Global-Locally Self-Attentive Dialogue State Tracker

This repository contains an implementation of the [Global-Locally Self-Attentive Dialogue State Tracker (GLAD)](https://arxiv.org/abs/1805.09655).
If you use this in your work, please cite the following

```
@inproceedings{ zhong2018global,
  title={ Global-Locally Self-Attentive Encoder for Dialogue State Tracking },
  author={ Zhong, Victor and Xiong, Caiming and Socher, Richard },
  booktitle={ ACL },
  year={ 2018 }
}
```

## Install dependencies

Using Docker

```
docker build -t glad:0.4 .
docker run --name embeddings -d vzhong/embeddings:0.0.5  # get the embeddings
env NV_GPU=0 nvidia-docker run --name glad -d -t --net host --volumes-from embeddings glad:0.4
```

If you do not want to build the Docker image, then run the following (you still need to have the CoreNLP server).

```
pip install -r requirements.txt
```

## Download and annotate data

This project uses Stanford CoreNLP to annotate the dataset.
In particular, we use the [Stanford NLP Stanza python interface](https://github.com/stanfordnlp/stanza).
To run the server, do

```
docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server
```

The first time you preprocess the data, we will [download word embeddings and character embeddings and put them into a SQLite database](https://github.com/vzhong/embeddings), which will be slow.
Subsequent runs will be much faster.

```
docker exec glad python preprocess_data.py
```

The raw data will be stored in `data/woz/raw` of the container.
The annotation results will be stored in `data/woz/ann` of the container.

If you do not want to build the Docker image, then run

```
python preprocess_data.py
```


## Train model

You can checkout the training options via `python train.py -h`.
By default, `train.py` will save checkpoints to `exp/glad/default`.

```
docker exec glad python train.py --gpu 0
```

You can attach to the container via `docker exec glad -it bin/bash` to look at what's inside or `docker cp glad /opt/glad/exp exp` to copy out the experiment results.

If you do not want to build the Docker image, then run

```
python train.py --gpu 0
```


## Evaluation

You can evaluate the model using

```
docker exec glad python evaluate.py --gpu 0 --split test exp/glad/default
```

You can also dump a predictions file by specifying the `--fout` flag.
In this case, the output will be a list of lists.
Each `i`th sublist is the set of predicted slot-value pairs for the `i`th turn.
Please see `evaluate.py` to see how to match up the turn predictions with the dialogues.

If you do not want to build the Docker image, then run

```
python evaluate.py --gpu 0 --split test exp/glad/default
```


## Contribution

Pull requests are welcome!
If you have any questions, please create an issue or contact the corresponding author at `victor <at> victorzhong <dot> com`.
