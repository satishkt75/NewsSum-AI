"""Human annotation support. Does not invent human annotations."""
import csv
import random


def export_blinded(sources, baseline, proposed, output, seed=42):
    if not (len(sources) == len(baseline) == len(proposed)):
        raise ValueError('Input lists must be aligned')
    rng = random.Random(seed)
    with open(output, 'w', encoding='utf8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['sample_id','source','summary_A','summary_B',
                                                   'A_factual','B_factual','A_relevance',
                                                   'B_relevance','A_coherence','B_coherence'])
        writer.writeheader()
        for i, (src, base, new) in enumerate(zip(sources,baseline,proposed)):
            if rng.random() < .5:
                base,new = new,base
            writer.writerow({'sample_id':i, 'source':src, 'summary_A':base, 'summary_B':new})
