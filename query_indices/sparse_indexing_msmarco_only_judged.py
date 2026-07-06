import pyterrier as pt

msmarco_dataset = pt.get_dataset('irds:msmarco-passage/train')
training_qrel = msmarco_dataset.get_qrels()
data_df = msmarco_dataset.get_topics()
data_df = data_df[data_df.qid.isin(training_qrel.qid.unique())]
iter_dict = dict(zip(data_df.qid, data_df['query']))

def _doc_generator():
    for _pair in iter_dict.items():
        yield {'docno': _pair[0], 'text': _pair[1]}

data_q = GeneratorLen(_doc_generator(), len(iter_dict))
indexer = pt.IterDictIndexer(f'./sparse', meta={'text': 256, 'docno': 64}, fields=True)

try:
    index_ref = indexer.index(data_q)
except:
    print('the query index alreay exists')