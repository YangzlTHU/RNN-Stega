# RNN-Stega

This code belongs to [RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks](https://ieeexplore.ieee.org/abstract/document/8470163/).

## Requirements

- python 2.7
- tensorflow = 1.0.0

## Prepare for Generating

- Download our corpus file [movie](https://drive.google.com/open?id=1Qnd3h5P0FWieTbH5MYZrq2j3K6JP1U4b) 
or [tweet](https://drive.google.com/open?id=1Z9ANQWt95EUaRgtCMFCVo2S5P8VafeEp) and put it in `./data/`
- Download pretrained models [movie model](https://drive.google.com/open?id=13F3Xt6zw8JYyzd-HlitXYknIPL_Hn_hA) or [tweet model](https://drive.google.com/open?id=1nDgoy6eE1aOWg9tRdn5Q4aIXhWfUfawU) and put it in `./models/movie/` or `./models/tweet/`
- Modify `./Config_movie.py` or `./Config_tweet.py` to adjust the hyperparameters

## Generate 

- movie dataset

```bash
python huffman_movie_v2.py 1 1
```

- tweet dataset

```bash
python huffman_tweet_v2.py 1 1
```

The args stand for bit and index.
