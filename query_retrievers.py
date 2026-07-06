#################
# qv retrievers #
#################

import pyterrier as pt
import pyterrier_dr

# sparse query index is shared by every pipeline
sparse_index = pt.IndexFactory.of(f'./query_indices/sparse')

# bm25 qv retriever
def get_bm25_q_retriever():
    bm25_query_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(sparse_index, wmodel="BM25", num_results=10) >> sparse_index.text_loader(["text"]) >> pt.rewrite.reset()
    return bm25_query_pipeline

# sbert qv retriever
def get_sbert_q_retriever():
    sbert_model = pyterrier_dr.SBertBiEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sbert_query_index = pyterrier_dr.FlexIndex('../get_res/sbert_msmarco_training_judged_query_index.flex')
    sbert_query_pipeline = sbert_model >> sbert_query_index.torch_retriever(qbatch=8, fp16=True, num_results=10)
    return sbert_query_pipeline >> sparse_index.text_loader(["text"])

# dragon qv retriever
def get_dragon_q_retriever():
    dragon_model = pyterrier_dr.SBertBiEncoder(model_name="facebook/dragon-plus-query-encoder")
    dragon_query_index = pyterrier_dr.FlexIndex('../get_res/dragon_msmarco_training_judged_query_index.flex')
    dragon_query_pipeline = dragon_model >> dragon_query_index.torch_retriever(qbatch=8, fp16=True, num_results=10)
    return dragon_query_pipeline >> sparse_index.text_loader(["text"])

# tct qv retriever
def get_tct_q_retriever():
    tct_model = pyterrier_dr.TctColBert()
    tct_query_index = pyterrier_dr.FlexIndex('../get_res/tct_msmarco_training_judged_query_index.flex')
    tct_query_pipeline = tct_model >> tct_query_index.torch_retriever(qbatch=8, fp16=True, num_results=10)
    
    return tct_query_pipeline >> sparse_index.text_loader(["text"])

# dragon qv retriever where queries were indexed as documents
def get_dragon_qasd_q_retriever():
    dragon_model = pyterrier_dr.SBertBiEncoder(model_name="facebook/dragon-plus-query-encoder")
    dragon_query_index = pyterrier_dr.FlexIndex('../get_res/dragon_qasd_msmarco_training_judged_query_index.flex')
    dragon_query_pipeline = dragon_model >> dragon_query_index.torch_retriever(qbatch=8, fp16=True, num_results=10)
    return dragon_query_pipeline >> sparse_index.text_loader(["text"])

# tct qv retriever where queries were indexed as documents
def get_tct_qasd_q_retriever():
    tct_model = pyterrier_dr.TctColBert()
    tct_query_index = pyterrier_dr.FlexIndex('../get_res/tct_qasd_msmarco_training_judged_query_index.flex')
    tct_query_pipeline = tct_model >> tct_query_index.torch_retriever(qbatch=8, fp16=True, num_results=10)
    
    return tct_query_pipeline >> sparse_index.text_loader(["text"])