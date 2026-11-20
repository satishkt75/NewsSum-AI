
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['cnn_dailymail','xsum','multi_news'])
    p.add_argument('--entity-vocab', required=True)
    args = p.parse_args()
    for variant in ('baseline','salience','faith','salience_faith'):
        for seed in (42,1234,2025):
            extra = f' --entity-vocab {args.entity_vocab}' if 'faith' in variant else ''
            print(f'python scripts/train.py --dataset {args.dataset} --variant {variant} --seed {seed}{extra}')

if __name__ == '__main__':
    main()
