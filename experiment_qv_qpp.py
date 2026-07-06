import pandas as pd
import numpy as np
import json
from qpp_methods.qpp import NQC_QPP
from qpp_methods.UEFramework import UEFramework
from scipy import stats

import pyterrier as pt

from pathlib import Path
import argparse
from tqdm import tqdm

info_dict = {'webis-touche2020': {'path': 'irds:beir/webis-touche2020/v2', 'meta': {'docno': 64, 'title': 100, 'text': 4096, 'url': 256, 'stance':16}},
             'trec_covid': {'path': 'irds:beir/trec-covid', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
             'dl_19': {'path': "irds:msmarco-passage/trec-dl-2019/judged", 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
             'dl_20': {'path': "irds:msmarco-passage/trec-dl-2020/judged", 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
             'dl_21': {'path': 'irds:msmarco-passage-v2/trec-dl-2021/judged', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
             'dl_22': {'path': 'irds:msmarco-passage-v2/trec-dl-2022/judged', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
            }

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='dl_19', choices=['dl_19', 'dl_20', 'dl_21', 'dl_22', 'webis-touche2020', 'trec_covid'])
    parser.add_argument("--retrieval", type=str, default='bm25_monoT5', choices=['bm25', 'bm25_monoT5', 'bm25_rankllama', 'tct', 'tct_monoT5', 'tct_rankllama'])
    parser.add_argument("--q_retriever", type=str, default='bm25', choices=['bm25', 'sbert', 'dragon', 'tct'])
    parser.add_argument("--hop_num", type=int, default=1, choices=[1, 2]) # 1-hop or 2-hop retrieved QVs are used in the experiment
    parser.add_argument("--k", type=int, default=1) # the number of QVs leveraged in QPP
    parser.add_argument("--p", type=int, default=-1) # p=-1 if it only uses retrieved QVs 
    parser.add_argument("--base_predictor", type=str, default='nqc')
    args = parser.parse_args()
    
    dataset = args.dataset_name
    retrieval = args.retrieval
    q_rtr = args.q_retriever
    hop_num = f'{args.hop_num}hop'
    k = args.k
    p = args.p
    base_predictor = args.base_predictor

    print('[debug]', q_rtr)

    output_dir = f'./exp_res/{dataset}_{retrieval}_{base_predictor}_{hop_num}_{q_rtr}_{k}.csv' if (p==-1) else f'./exp_res/{dataset}_{retrieval}_{base_predictor}_{p}shot_{hop_num}_{q_rtr}_{k}.csv'
    if Path(output_dir).exists():
        print("File exists", output_dir)
        raise RuntimeError("don't need to continue")
    else:
        pass
    
    dataset_obj = pt.get_dataset(info_dict[dataset]['path'])
    
    # load index
    
    if dataset in ['dl_19', 'dl_20']:
        index_path = '/doc_indices/msmarco-passage.terrier/'
    elif(dataset in ['dl_21', 'dl_22']):
        index_path = '/doc_indices/msmarco-passage-v2-dedup.terrier'
    else:
        index_path = f'./doc_indices/{dataset}/'
    
    index_ref = pt.terrier.TerrierIndex(index_path).index_ref()
    sparse_index = pt.IndexFactory.of(index_ref, memory=['inverted', 'lexicon'])
    bm25_pipeline = pt.terrier.Retriever(index_ref, wmodel="BM25") % 100
    
    if(p < 0):
        qv_df = pd.read_csv(f'./qv_res/reranked_{dataset}_{q_rtr}_{hop_num}.csv')
    else:
        qv_df = pd.read_csv(f'./qv_res/reranked_{dataset}_{p}shot_{q_rtr}_{hop_num}.csv')
    qv_df.qid = qv_df.qid.astype('str')
    
    #read res
    res = pd.read_csv(f'./res/{dataset}_{retrieval}.csv')
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    if(dataset in ['dl_21', 'dl_22']):
        pass
    else:
        if('monoT5' in retrieval):
            res['score'] = res['score'].apply(np.exp)
        elif('rankllama' in retrieval):
            res['score'] = res['score'].apply(sigmoid)
        
    res.qid = res.qid.astype('str')
    
    with open(f'./eval_res/perquery_{dataset}_{retrieval}.json') as f:
        eval_res = json.load(f)
        eval_df = pd.DataFrame.from_dict(eval_res, orient='index').reset_index().rename(columns={"index": "qid"})
    
    # base qpp model
    
    nqc = NQC_QPP(index_path=index_path)

    try:
        if(base_predictor=='nqc'):
            base_qpp = nqc
        elif(base_predictor=='uef'):
            base_qpp = UEFramework(nqc)
    except:
        print("Predictor has not been supported!")
    
    qv_qpp_df_content = []
    
    def combine_qv_qpps(x):
        # print('[debug]', x)
        x = x.dropna()
        if(x.shape[0]==0):
            return -1
        sum_rbo = x.rbo.sum()
    
        if(sum_rbo > 0):
            return x.apply(lambda _x: _x['rbo']*_x['ref_est'], axis=1).sum() / sum_rbo
        else:
            return x['ref_est'].mean()
    
    for qid in tqdm(res.qid.unique()):
        # print(qid)
        estimate = base_qpp.compute(res=res, qid=qid, topk=50)
        
    
        qv_df_for_qid = qv_df[qv_df.qid==qid][['rqid', 'rqText', 'rank', 'score']].copy()
        qv_df_for_qid = qv_df_for_qid.rename(columns={'rqid': 'qid', 'rqText': 'query', 'rank': 'rbo_rank', 'score': 'rbo'}).query('rbo_rank<@k')
        temp_qv_res = bm25_pipeline(qv_df_for_qid)
        temp_qv_res['score'] = temp_qv_res['score'].apply(lambda x: x)
    
        temp_dict = {}
        for rqid in qv_df_for_qid.qid:
            temp_dict.update({rqid: base_qpp.compute(temp_qv_res, rqid, topk=50)})
    
        qv_df_for_qid['ref_est'] = qv_df_for_qid.qid.apply(lambda x: temp_dict[x])
        
        qv_part = combine_qv_qpps(qv_df_for_qid)
        if(qv_part == -1):
            qv_part = estimate
        qv_qpp_df_content.append([qid, estimate, qv_part])
        
    
    qv_qpp_df = pd.DataFrame(qv_qpp_df_content, columns=['qid', 'estimate', 'qv_est'])
    eval_qv_qpp_df = eval_df.merge(qv_qpp_df, on=['qid'])
    # print('[debug]', eval_qv_qpp_df)
    cols = ['estimate', 'qv_est']
    
    df_content = []
    
    for coeff in np.arange(0, 1.001, 0.1):
        eval_qv_qpp_df['final_est'] = eval_qv_qpp_df.apply(lambda x: coeff*x['estimate']+(1-coeff)*x['qv_est'], axis=1)
        tau = stats.kendalltau(eval_qv_qpp_df['AP(rel=2)@100'], eval_qv_qpp_df['final_est'])
        tau_ndcg = stats.kendalltau(eval_qv_qpp_df['nDCG@10'], eval_qv_qpp_df['final_est'])
        df_content.append([coeff, tau[0], tau_ndcg[0], k])
    
    pd.DataFrame(df_content, columns=['lambda', 'tau_ap', 'tau_ndcg', 'k']).to_csv(output_dir, index=False)