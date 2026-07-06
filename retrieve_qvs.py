#######################################################
# Retrieving QVs using various query retrievers.      #
# The results will be re-ranked by RBO automatically. #
#######################################################

import pyterrier as pt
import pyterrier_dr
import pyterrier_alpha as pta
from pyterrier.model import add_ranks

from query_retrievers import *
import hopper

import pandas as pd
from tqdm import tqdm
import time
import argparse
import sys

path_dict = {'webis-touche2020': 'irds:beir/webis-touche2020/v2',
             'trec_covid': 'irds:beir/trec-covid',
             'dl_19': "irds:msmarco-passage/trec-dl-2019/judged",
             'dl_20': "irds:msmarco-passage/trec-dl-2020/judged",
             'dl_21': "irds:msmarco-passage-v2/trec-dl-2021/judged",
             'dl_22': "irds:msmarco-passage-v2/trec-dl-2022/judged",
            }

# tool function for converting format
def transform_qv_df(_res):

    _res['rqid'] = _res[['qid', 'docno']].apply(lambda x: f'{x["qid"]}_{x["docno"]}', axis=1)
    _res = _res.rename(columns={'query': 'qText', 'text': 'rqText'})
    _res = _res[['qid', 'qText', 'rqid', 'rqText']]

    return _res

#########
# start #
#########

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='dl_19', choices=list(path_dict.keys()))
parser.add_argument("--q_retriever", type=str, default='bm25', choices=['bm25', 'sbert', 'dragon', 'tct', 'dragon_qasd', 'tct_qasd'])
parser.add_argument("--hop_num", type=int, default=1, choices=[1, 2])
args = parser.parse_args()

exp_name = args.dataset_name
q_retriever = args.q_retriever
hop_num = args.hop_num

print(f"[progress] We are doing {hop_num}_hop QV retrieval with {q_retriever} as query retriever for {exp_name} queries.")

# loading queries, corresponding indices, and bm25 doc retrieval pipeline
test_queries = pt.get_dataset(path_dict[exp_name]).get_topics('text')
msmarco_dataset = pt.get_dataset('irds:msmarco-passage/train')

if(exp_name in ['dl_19', 'dl_20']):
    tgt_dataset_doc_index = pt.IndexFactory.of('/doc_indices/msmarco-passage.terrier/')
elif(exp_name in ['dl_21', 'dl_22']):
    tgt_dataset_doc_index = pt.IndexFactory.of('/doc_indices/msmarco-passage-v2-dedup.terrier')
else:
    tgt_dataset_doc_index = pt.IndexFactory.of(f'./doc_indices/{exp_name}')

bm25_doc_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(tgt_dataset_doc_index, wmodel="BM25", verbose=True, num_results=20) % 20 >> tgt_dataset_doc_index.text_loader(["text"]) >> pt.rewrite.reset() # we only need top-20 for RBO

print('[progress] Finished preparation.')

###############################
# instantiating qv retrievers #
###############################

first_hop_retriever = getattr(sys.modules[__name__], f"get_{q_retriever}_q_retriever")()

print('[progress] Query retriever loaded.')

######################
# retrieve 1-hop qvs #
######################

first_pipeline = first_hop_retriever % 10 >> pt.apply.generic(transform_qv_df)
qvs = first_pipeline(test_queries)

print('[progress] QVs have been retrieved.')

#######################
# extend to 2-hop qvs #
#######################
if(hop_num == 2):

    second_hop_retriever = get_bm25_q_retriever()  
    second_pipeline = first_pipeline >> pt.apply.generic(lambda x: hopper.second_hop(x, msmarco_dataset, second_hop_retriever))
    qvs = pd.concat([qvs, second_pipeline(test_queries)]).drop_duplicates()

    print(f'[progress] QVs have been extended.')   

########################
# reranking qvs by RBO #
########################

# retrieve documents for test queries and qvs
orig_res_df = bm25_doc_pipeline(test_queries)
qvs_res_df = bm25_doc_pipeline(qvs[['rqid', 'rqText']].rename(columns={'rqText': 'query', 'rqid': 'qid'}))

qvs_res_df['orig_qid'] = qvs_res_df['qid'].apply(lambda x: x.split('_')[0])
qvs_rbo_rerank_df_content = []

print('[progress] Computing RBO ....')

for qid, qText in tqdm(test_queries[['qid', 'query']].values):
    res_0 = orig_res_df[orig_res_df.qid==qid]
    res_1 = qvs_res_df[qvs_res_df.orig_qid==qid] # query with all its retrieved qvs
    
    for rqid in res_1.qid.unique():
        # for each qv
        temp_res = res_1[res_1.qid==rqid].copy() # get its retrieval results
        temp_res = temp_res.drop(columns=['qid'])
        temp_res = temp_res.rename(columns={'orig_qid': 'qid'})

        rqText = temp_res['query'].values[0]
        rbo_value = pta.rbo(res_0[res_0['rank']<20], temp_res[temp_res['rank']<20])
        qvs_rbo_rerank_df_content.append([qid, qText, rqid, rqText, list(rbo_value)[0][1]])

qvs_rbo_rerank_df = pd.DataFrame(qvs_rbo_rerank_df_content, columns=['qid', 'qText', 'rqid', 'rqText', 'score'])
qvs_rbo_rerank_df = add_ranks(qvs_rbo_rerank_df)
qvs_rbo_rerank_df = qvs_rbo_rerank_df[qvs_rbo_rerank_df['rank']<10]

print('[progress] QVs have been re-ranked.')

# save results
qvs_rbo_rerank_df.to_csv(f'./qv_res/reranked_{exp_name}_{q_retriever}_{hop_num}hop.csv', index=False)
print('[progress] QVs have been stored.')
