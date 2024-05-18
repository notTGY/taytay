# TayTay v2

Learned on songs from https://github.com/shaynak/taylor-swift-lyrics/blob/main/songs.csv

## Fun facts

The only major change to architecture from v1
is BPE tokenizer. This time I actually read
Chinchilla paper. I know, that I have 800k tokens,
I want to train the model for approximately
100 epochs. So I went for 1.2M params and left my
laptop running during watching
"Fast & Furious: Hobbs & Shaw", which is an
amazing movie, by the way.
(For the reference, 100 epochs take ~2 hours for me)

During evaluation I asked model to continue the same
prompt of original model: "a". And I also included
probability of completion. Without beam search it
looks dramatic sometimes, but it is interesting to
see "Dunning–Kruger effect of Neural network", where 
it completes sequence the same way, but is less sure 
about it's correctness after epochs pass by.

Tokenizer made model write more coherent text and
even during training it gave me perfectly fine
Taylor-Swift-like lyrics "ah,out,out,out,out,"
Something you could really witness in her works.
After exactly 100 epochs the result was:
"all they, is" with 6.4% certainty. Decent.
You can play with this exact checkpoint
"taytay-hobbs.pt".


## Conclusion

TayTay v2 is a real piece.

Due to my laziness there would be no comparison to
the original model. Also, during tokenizator
adaptation I have rewritten some code of the model,
so I guess it was flawed, which gives not much
assurance current version is not. Parameters
grew from 120k to 1.2M (which is adorable, because
this resembles growth of
GPT-1 => GPT-2 => GPT-3 => GPT-4 by a factor of 10)

