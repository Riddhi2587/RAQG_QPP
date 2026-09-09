####################################
# Re-rank the generated QVs by RBO #
####################################

import pandas as pd
from pyterrier.model import add_ranks
import json
import pyterrier as pt
import pyterrier_alpha as pta
import argparse
from pathlib import Path
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='dl_19',
                         help="Dataset key used for output filenames; must match the --dataset_name used "
                              "for the earlier retrieve_qvs.py/generate_qvs.py runs.")
    parser.add_argument("--q_retriever", type=str, default='bm25', choices=['bm25', 'sbert', 'dragon', 'tct', 'dragon_qasd', 'tct_qasd'])
    parser.add_argument("--hop_num", type=int, default=1, choices=[1, 2])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--doc_index_path", type=str, default=None,
                         help="Optional path to a prebuilt Terrier index of the target corpus, overriding the "
                              "hardcoded default paths (which assume a specific server layout).")
    args = parser.parse_args()

    dataset = args.dataset_name
    q_rtr = args.q_retriever
    hop_num = args.hop_num
    p = args.p

    output_dir = f'./qv_res/reranked_{dataset}_{p}shot_{q_rtr}_{hop_num}hop_paraphrase.csv'
    if Path(output_dir).exists():
        print("File exists", output_dir)
        raise RuntimeError("don't need to continue")
    else:
        print("Starting....")
        pass
    
    orig_res_df = pd.read_csv(f'./res/{dataset}_bm25.csv')
    orig_res_df.docno = orig_res_df.docno.astype('str')
    orig_res_df.qid = orig_res_df.qid.astype('str')

    if args.doc_index_path is not None:
        index_path = args.doc_index_path
    elif(dataset in ['dl_19', 'dl_20']):
        index_path = '/doc_indices/msmarco-passage.terrier/'
    elif(dataset in ['dl_21', 'dl_22']):
        index_path = '/doc_indices/msmarco-passage-v2-dedup.terrier'
    else:
        index_path = f'./doc_indices/{dataset}/'

    print("Building retrieval pipeline")
    # index_ref = pt.terrier.TerrierIndex(index_path).index_ref()
    # sparse_index = pt.IndexFactory.of(index_ref, memory=['inverted', 'lexicon'])
    # bm25_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(index_ref, wmodel="BM25") % 20
    
    sparse_index = pt.IndexFactory.of(index_path)
    bm25_pipeline = pt.rewrite.tokenise() >> pt.terrier.Retriever(sparse_index, wmodel="BM25", controls={'bm25.k_1': '0.9', 'bm25.b': '0.4'}) % 20 >> pt.rewrite.reset()
    
    # >> dataset_obj.text_loader(["text"])

    print("Reading QVs", f'./gen_qv_res/{dataset}_{p}shot_{hop_num}hop_{q_rtr}_paraphrase_qvs.json')
    with open(f'./gen_qv_res/{dataset}_{p}shot_{hop_num}hop_{q_rtr}_paraphrase_qvs.json') as f:
        unranked_qvs = json.load(f)

    print("converting to csv")
    qvs_res_content = []
    for qid, details in unranked_qvs.items():
        # print(qid)
        qText = details['query']
        # print(qid, qText)
        for i, (key, qv_details) in enumerate(details['gen_qvs'].items()):
            qvs_res_content.append([qid, qText, f'{qid}_gen{i}', qv_details])

    # print('[debug]--point 0')
    qvs_res = pd.DataFrame(qvs_res_content, columns=['qid', 'qText', 'rqid', 'rqText'])
    print('[debug] start retrieving')
    qvs_rtr_res = 0
    for i in tqdm(range(0, qvs_res.shape[0], 5)):
        temp_qvs_rtr_res = bm25_pipeline(qvs_res.iloc[i: i+5][['rqid', 'rqText']].rename(columns={'rqid': 'qid', 'rqText': 'query'}))
        if(type(qvs_rtr_res)==int):
            qvs_rtr_res = temp_qvs_rtr_res
        else:
            qvs_rtr_res = pd.concat([qvs_rtr_res, temp_qvs_rtr_res])

    print('[debug]', qvs_rtr_res)
    qvs_rtr_res['orig_qid'] = qvs_rtr_res['qid'].apply(lambda x: x.split('_')[0])
    qvs_rbo_rerank_df_content = []

    print("Start reranking")
    
    for qid in orig_res_df['qid'].unique():
        print('[debug]', qid)
        qText = orig_res_df.query('qid==@qid')['query'].values[0]
        res_0 = orig_res_df[orig_res_df.qid==qid]
        res_1 = qvs_rtr_res[qvs_rtr_res.orig_qid==qid]
        
        for rqid in res_1.qid.unique():
            # print(rqid)
            res_11 = res_1[res_1.qid==rqid].copy()
            res_11 = res_11.drop(columns=['qid'])
            res_11 = res_11.rename(columns={'orig_qid': 'qid'})
    
            rqText = res_11['query'].values[0]
            rbo_value = pta.rbo(res_0[res_0['rank']<20], res_11[res_11['rank']<20])
            qvs_rbo_rerank_df_content.append([qid, qText, rqid, rqText, list(rbo_value)[0][1]])

    qvs_rbo_rerank_df = pd.DataFrame(qvs_rbo_rerank_df_content, columns=['qid', 'qText', 'rqid', 'rqText', 'score'])
    qvs_rbo_rerank_df = add_ranks(qvs_rbo_rerank_df)
    qvs_rbo_rerank_df = qvs_rbo_rerank_df[qvs_rbo_rerank_df['rank']<10]
    qvs_rbo_rerank_df.to_csv(output_dir, index=False)