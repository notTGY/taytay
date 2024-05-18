from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=128, special_tokens=["~"])
tokenizer.train(["songs.txt"], trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

sentence = "i love you baby."

encoding = tokenizer.encode(sentence)

tokenizer.decoder = decoders.ByteLevel()
print(tokenizer.decode(encoding.ids))

tokenizer.save("tokenizer.json")

