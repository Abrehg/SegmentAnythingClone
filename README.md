# Segment Anything Clone

Pet project that replicates Meta's Segment Anything Model (https://segment-anything.com/demo#) and its paper by (Berg et al.)

However instead of physical markers to mark where the object is, this model uses a text prompt to decide where to place the mask

## How the model works

This model implements two transformer encoders based on the paper Attention is All You Need (Gomez et al.). 

One encoder encodes the input image by converting each 16x16 patch in the image into a vector input using a pre-trained patch model and then passing it into a typical transformer encoder

The other encoder encodes the text by converting each string into a vector of vectors using GloVe and then passing it into a typical transformer encoder

Finally, a decoder loosely based on the standard transformer decoder takes both encodings and outputs a mask of similar size to the input image with pixel values between 0 and 1 to show where the masked object is

## Dependancies

This model uses the following dependancies:
- conll2003 dataset to train the text encoder, 
- GloVe embeddings in the form 840B words with 300d for text to vec (downloadable from https://nlp.stanford.edu/projects/glove/)
- ADE20K dataset to train main model

## State of the project

Basic architecture developed, just need to train the model

Started 6/30/23, Finished --/--/24
