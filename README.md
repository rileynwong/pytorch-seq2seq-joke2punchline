# PyTorch Seq2Seq / Joke2Punchline / Punchline2Joke
PyTorch implementation of the [sequence-to-sequence](https://arxiv.org/abs/1409.3215) neural network model with [attention](https://arxiv.org/abs/1409.0473). Includes pre-trained models for:
- joke2punchline: Given a question-format joke, output a generated punchline.
- punchline2joke: Given a punchline, output a generated question-format joke.
- eng<>fra: Given a French sentence, output its English translation, or vice versa.

See blog posts on [seq2seq language translation](https://www.rileynwong.com/blog/2019/4/3/implementing-a-seq2seq-neural-network-with-attention-for-machine-translation-from-scratch-using-pytorch) and for [joke2punchline and punchline2joke](https://www.rileynwong.com/blog/2019/4/12/joke2punchline-punchline2joke-using-a-seq2seq-neural-network-to-translate-between-jokes-and-punchlines) (contains more examples!).

## Joke2Punchline
Given a question-format joke, output a generated punchline.

```
> what do you call an unpredictable chef ?
< ouch .

> what do you call a pile of pillowcases ?
< screw music 

> why was the sun hospitalized ?
< because he was sitting on me . 

> who s there ?
< in the dictionary . 

> what is red and bad for your teeth ?
< a a gummy bear 
```

## Punchline2Joke
Given a punchline, output a generated question-format joke.

```
> watermelon concentrate
< when do you stop at green and go at the orange juice factory ? 

> cool space
< what do you call an alligator in a vest with a scoop of ice cream ? 

> the impossible !
< what did the worker say when he swam into the wall ? 

> both !
< what do you call a ghosts mom and dad ? 

> one two three four
< what did the buffalo say to the bartender ? 
```

## Eng2Fra
Given a French sentence, output the English translation.
```
> je n appartiens pas a ce monde .
= i m not from this world .
< i m not from this world .

> je suis impatient de la prochaine fois .
= i m looking forward to the next time .
< i m looking forward to the next time . 

> tu es sauve .
= you re safe .
< you re safe . 

> vous etes a nouveau de retour .
= you re back again .
< you re back again . 

> il n est pas marie .
= he s not married .
< he s not married . 
```

## Project Overview
- Scripts:
  - `joke2punchline.py`: Generate punchlines given jokes.
  - `punchline2joke.py`: Generate jokes given punchlines.
  - `seq2seq.py`: Train or translate English<>French sentences.
- Pre-trained models:
  - `models/` directory:
    - Contains trained encoders and decoders for joke2punchline, punchline2joke, and eng<>french. 
- Data:
  - `jokes_data/jokes.tsv`: Tab-separated file with question-answer format jokes.
  - `eng_translation_data/`: Contains tab-separated text files with English-French sentence pairs.

## Credits
- PyTorch [seq2seq](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) tutorial
- Clean question-answer format jokes: [1](http://www.jokes4us.com/miscellaneousjokes/cleanjokes.html) [2](http://www.tensionnot.com/jokes/one_liner_jokes/funny_questions_and_answers) [3](https://www.quickfunnyjokes.com/cheesy.html)
- Language pairs data from the [Tatoeba Project](https://www.manythings.org/anki/)
