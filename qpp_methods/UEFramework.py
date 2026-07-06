import pyterrier as pt
import pandas as pd
import numpy as np
from pyterrier.model import add_ranks
from qpp import BaseQPP
from RelevanceModels import RelevanceModelConditional

def compute_rank_dist(rank_df_reranked, rank_df_orig):

    tempt_df = pd.merge(rank_df_reranked[['qid', 'docid', 'rank', 'orig_score']], rank_df_orig[['docid', 'orig_score']], on=['docid', 'orig_score'])
    tempt_df = tempt_df.rename(columns={"orig_score": "score", "rank": "rank_1"})
    tempt_df = add_ranks(tempt_df)
    tempt_df = tempt_df.rename(columns={"rank": "rank_0"})
    tempt_df['sq_diff'] = pow(tempt_df['rank_0'] - tempt_df['rank_1'], 2)
    # print(tempt_df, pow(tempt_df.sq_diff.mean(), 1/2))

    avg_shift = pow(tempt_df.sq_diff.mean(), 1/2)
    return avg_shift

class Shuffler(pt.Transformer):
    def transform(self, run):
        run = run.sample(frac = 1)
        run = run.rename(columns={"score": "orig_score"})
        run = run.drop(columns = ["rank"])
        run["score"] = list(range(len(run) + 1, 1, -1))
        run = add_ranks(run)
        return run

class UEFramework(BaseQPP):

    def __init__(self, qpp_model, index_path='/mnt/indices/msmarco-passage.terrier/'):
        super(UEFramework, self).__init__(index_path)
        np.random.seed(1024)
        self.QPP = qpp_model # this is a method for now
        self.shuffler = Shuffler()
        self.NUM_SAMPLES = 10

    def sample_docs(self, res, query_df):
        sample_pipeline = (pt.Transformer.from_df(res) % self.pool_for_k) >> self.shuffler % self.k_for_rlm
        return sample_pipeline(query_df)

    def compute(self, res, qid, k: int, topk: int):

        self.k_for_rlm = k
        self.pool_for_k = 3*k

        try:
            res.qid = res.qid.astype('str')
            qText = res[res.qid==str(qid)]['query'].values[0]
        except:
            print('This res doesn\'t contain this qid.')
        query_df = pd.DataFrame([[str(qid), qText]], columns=['qid', 'query'])

        rank_dist_list = []

        for _i in range(self.NUM_SAMPLES):
            # print('[round]', _i)
            sampled_df_for_q = self.sample_docs(res, query_df)
            # print(sampled_df_for_q)

            rlm = RelevanceModelConditional(self.index, query_df, sampled_df_for_q, k=self.k_for_rlm)
            rlm.compute_feedback_weights()
            # print('[debug]', rlm.feedback_weights)
            reranked_sample = rlm.rerank_docs()
            # print('[debug]', reranked_sample)

            rank_dist = compute_rank_dist(reranked_sample, sampled_df_for_q)
            rank_dist_list.append(rank_dist)

        base_estimation = self.QPP.compute(res=res, qid=qid, topk=topk)
        uef_coeff = self.NUM_SAMPLES/np.mean(rank_dist_list)

        return uef_coeff*base_estimation