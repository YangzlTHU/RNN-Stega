# RNN-Stega

This code belongs to [RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks](https://ieeexplore.ieee.org/abstract/document/8470163/).

## Requirements

- python 2
- tensorflow = 1.0.0

## Prepare for Generating

- Download our corpus file [movie](https://drive.google.com/file/d/1LP4ZIZsHDRf2ZgiMIu2EAIex_iC5WGFM/view?usp=sharing) 
or [tweet](https://drive.google.com/file/d/12YDuBm29TPkgB-zOpuBBRBBELdjb0uNb/view?usp=sharing) and put it in `./data/`
- Download pretrained models [movie-model]() or [tweet-model]() and put it in `./models/movie/` or `./models/tweet/`
- Modify `./Config_movie.py` or `./Config_tweet.py` to adjust the hyperparameters

## Generate 

- movie dataset

```bash
python huffman_movie_v2.py
```

- tweet dataset

```bash
python huffman_tweet_v2.py
```
