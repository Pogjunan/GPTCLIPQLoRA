import argparse, sentencepiece as spm, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='plain text file for tokenizer training')
    ap.add_argument('--model_prefix', type=str, default='tokenizer/spm')
    ap.add_argument('--vocab_size', type=int, default=50000)
    ap.add_argument('--character_coverage', type=float, default=1.0)
    ap.add_argument('--model_type', type=str, default='bpe', choices=['bpe','unigram'])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        input_sentence_size=2000000,
        shuffle_input_sentence=True
    )
    print(f"Saved {args.model_prefix}.model and .vocab")

if __name__ == '__main__':
    main()
