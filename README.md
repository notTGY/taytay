# AI neural network model

Learned on songs from https://github.com/shaynak/taylor-swift-lyrics/blob/main/songs.csv

For some reason model really likes char `w`
sample output:
```
and w w t we we the we the were the w the wand we we we we the the we the w we the the w we w
we w the we w ound the we t we w w we w w wund the we we we t we w w
```

Using words was too inefficient, model didn't learn anything from that data.

My GPU can handle max `721600` param model training :(
So if you want to see results, give me decent GPU.
Until then no paper.

(To prove that I haven't messed up I tried training SOTA model\* of comparable size on the same data
and got approximately the same results (loss ~8 on 16 seq length batches). So TayTay should be doable)

* Gemma has special license, but I haven't used it in any way to generate text or in any other way
except of being baseline for performance.
