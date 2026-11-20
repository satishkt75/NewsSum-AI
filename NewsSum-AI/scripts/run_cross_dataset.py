
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cnn-checkpoint', required=True)
    p.add_argument('--xsum-checkpoint', required=True)
    args = p.parse_args()
    print(f'python scripts/batch_predict.py --checkpoint {args.cnn_checkpoint} --dataset xsum --output outputs/cnn_to_xsum.jsonl')
    print(f'python scripts/batch_predict.py --checkpoint {args.xsum_checkpoint} --dataset cnn_dailymail --output outputs/xsum_to_cnn.jsonl')

if __name__ == '__main__':
    main()
