import pyterrier as pt
import pyterrier_rag
import pyterrier_dr
from pyterrier.utils import GeneratorLen

model = pyterrier_dr.TctColBert()

data_df = pt.get_dataset('irds:msmarco-passage/train/judged').get_topics()

iter_dict = dict(zip(data_df.qid, data_df['query']))

def _doc_generator():
    for _pair in iter_dict.items():
        yield {'qid': _pair[0], 'query': _pair[1]}

data_q = GeneratorLen(_doc_generator(), len(iter_dict))

index_path = "./tct_msmarco_training_judged_query_index.flex"
tct_msmarco_q_index = pyterrier_dr.FlexIndex(index_path)
pipeline_q = (model.query_encoder() >> pt.apply.generic(lambda x: x.rename(columns={'qid': 'docno', 'query_vec': 'doc_vec'})) >> tct_msmarco_q_index)

pipeline_q.index(data_q)