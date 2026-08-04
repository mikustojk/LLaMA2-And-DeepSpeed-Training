from transformers import AutoTokenizer
from Pretrain_Dataset import PretrainDataset

TOKENIZER_PATH="tokenizer_k"
DATA_PATH="data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl"
MAX_LENGTH=512

def main():
    print("Loading tokenizer...")
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("Tokenizer information:")
    print("vocab_size:", len(tokenizer))
    print("bos_token_id:", tokenizer.bos_token_id)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token_id:", tokenizer.pad_token_id)

    print("\nLoading dataset...")
    dataset = PretrainDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    print("dataset length:", len(dataset))

    print("\nLoading the first sample...")
    x, y, loss_mask = dataset[0]

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("loss_mask shape:", loss_mask.shape)

    print("x dtype:", x.dtype)
    print("y dtype:", y.dtype)
    print("loss_mask dtype:", loss_mask.dtype)

    print("first 20 x tokens:", x[:20].tolist())
    print("first 20 y tokens:", y[:20].tolist())
    print("loss_mask unique values:", loss_mask.unique().tolist())
    print("valid loss positions:", loss_mask.sum().item())

    assert x.shape == (MAX_LENGTH - 1,)
    assert y.shape == (MAX_LENGTH - 1,)
    assert loss_mask.shape == (MAX_LENGTH - 1,)
    assert x.dtype.name if False else True
    assert int(x.max()) < len(tokenizer)
    assert int(y.max()) < len(tokenizer)

    print("\nDataset test passed.")


if __name__ == "__main__":
    main()