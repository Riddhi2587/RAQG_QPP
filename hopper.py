# tool function for second hop qv retrieval
# _dataset should contain qrels

import pyterrier as pt

def second_hop(res, dataset, rtr_model):

    # get the qrel_df
    qrel_df = dataset.get_qrels()
    
    # get relevant documents (only take one per query)
    res['docno'] = res['rqid'].apply(lambda x: qrel_df[qrel_df.qid==x.split('_')[1]].docno.values[0])
    res = pt.text.get_text(dataset, 'text')(res)

    # give each of the pseudo queries an rqid
    res['rqid'] = res.apply(lambda x: f'{x["qid"]}_{x["docno"]}', axis=1)
    res = res.rename(columns={'text': 'query', 'rqid': 'qid', 'qid': 'orig_qid', 'qText': 'orig_query'})
    res = res.drop_duplicates()

    # retrieve qvs using those pseudo queries, take top 5 for each of them
    res = (rtr_model % 5)(res[['qid', 'query', 'orig_query']])
    res['qid'] = res['qid'].apply(lambda x: x.split('_')[0]) # restore the real qid
    res['rqid'] = res[['qid', 'docno']].apply(lambda x: f'{x["qid"]}_{x["docno"]}', axis=1) # second-hop qvs' ids are orig_qid+qv's_id
    res = res.rename(columns={'text': 'rqText', 'orig_query': 'qText'})
    
    res = res[['qid', 'qText', 'rqid', 'rqText']]
    res = res.drop_duplicates()

    return res
