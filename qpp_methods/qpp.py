import pyterrier as pt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math

from abc import ABC, abstractmethod
import pyterrier as pt
import numpy as np

class BaseQPP(ABC):
    def __init__(self, index_path='/mnt/indices/msmarco-passage.terrier/'):
        # if not pt.java.started():
        #     pt.java.init()

        # index_ref = pt.IndexRef.of(index_path)
        # self.index = pt.IndexFactory.of(index_ref)
        self.index = pt.terrier.TerrierIndex(index_path)
        self.DOC_NUM = self.index.collection_statistics().numberOfDocuments
        self.set_stopwords()

    @abstractmethod
    def compute(self, res, qid, **kwargs):
        """
        Compute a scalar QPP estimate for a single query.
        """
        pass

    def transform(self, res, **kwargs):
        output_content = []
        for qid in res.qid.unique():
            estimate = self.compute(res, qid, **kwargs)
            output_content.append([qid, estimate])

        return pd.DataFrame(output_content, columns=['qid', 'qpp_estimate'])

    # ---------- shared utility ----------

    def get_corpus_document_frequency(self, token):
        meta = self.index.meta_index()
        lexicon = self.index.lexicon()

        try:
            if(token in self.stopwords):
                d_freq = self.DOC_NUM
            else:
                d_freq = lexicon[token].getDocumentFrequency()
        except KeyError:
            d_freq = 1

        return d_freq
        
    def get_max_idf_query(self, qid, qText):
        stemmer = pt.TerrierStemmer.porter
        # print('[debug]', pt.rewrite.tokenise().search(qText)['query'].values[0])
        qText = pt.rewrite.tokenise().search(qText)['query'].values[0]

        dfs = []
        stemmed_ts = []
        for token in qText.split():
            if(token in self.stopwords):
                continue
            t = stemmer.stem(token)
            stemmed_ts.append(t)
            d_freq = self.get_corpus_document_frequency(t)
            dfs.append(d_freq)

        dfs = np.array(dfs)
        # print('[debug]', dfs, stemmed_ts, np.log(self.DOC_NUM/dfs))
        # return np.max(np.log((self.DOC_NUM - dfs + 0.5) / (dfs + 0.5))) # it should be it
        # but keep the same as Debasis' code:
        return np.max(np.log(self.DOC_NUM/dfs))

    def set_stopwords(self, file_path="./stopword-list.txt"):

        self.stopwords = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # strip whitespace & skip empty/comment lines
                word = line.strip()
                if word:
                    self.stopwords.append(word)
        
        # print(stopwords)
        print(f"Loaded {len(self.stopwords)} stopwords.")

class NQC_QPP(BaseQPP):

    def compute(self, res, qid, topk=100):
        res.qid = res.qid.astype('str')
        res_q = res[(res.qid == str(qid))]
        res_q = res_q.sort_values(by=['score'], ascending=False)
        res_q = res_q.iloc[:topk]
        scores_topk = np.array(res_q.score.values)
        var_value = np.var(scores_topk)

        try:
            qText = res[res.qid == str(qid)]['query'].values[0]
            # print(f'[debug] {qid}, {qText}')
        except:
            print('The res file doesn\'t contains this qid.')
            return nan
            
        max_idf = self.get_max_idf_query(qid, qText)
        # print('[debug]', qText, max_idf)
        nqc = var_value * max_idf
        # nqc = var_value

        return nqc

class APairRatio_QPP(BaseQPP):

    def compute(self, res, qid, _index):

        ratios = []
        s1 = 0.1
        s2 = 0.2
    
        s = min(50, len(res[res.qid==qid]))
            
        doc_df = res[(res.qid==qid)&(res['rank']<s)]
        doc_embs = _index.vec_loader()(doc_df)['doc_vec'].values
        emb_cluster = np.vstack(doc_embs)
        sim_mat = cosine_similarity(emb_cluster, dense_output=True)
    
        scores = res[(res.qid==qid)&(res['rank']<s)]['score'].values
        scores = np.expand_dims(scores, axis=1)
        score_m_mat = np.dot(scores, scores.T)
    
        sim_mat = sim_mat@score_m_mat
    
        sim_mat_top = sim_mat[:math.ceil(s1*s), :math.ceil(s1*s)].flatten()
        mean_top = np.mean(sim_mat_top)
    
        sim_mat_tail = sim_mat[int(s2*s):, int(s2*s):].flatten()
        mean_tail = np.mean(sim_mat_tail)
            
        prediction = mean_top/mean_tail
        
        return prediction.item()


class Spatial_QPP(BaseQPP):

    def compute(self, res, qid, topk, q_encoder, _index):
        ratios = []
        query_emb = q_encoder(res[res.qid==qid][['qid', 'query']].iloc[:1])
        query_emb = query_emb['query_vec'].values[0]
        emb_cluster = np.expand_dims(query_emb, axis=0)
            
        doc_df = res[(res.qid==qid)&(res['rank']<topk)]
        doc_embs = _index.vec_loader()(doc_df)['doc_vec'].values
        for e in doc_embs:
                # print(e.shape)
            emb_cluster = np.append(emb_cluster, np.expand_dims(e, axis=0), axis=0)
        max_emb = np.max(emb_cluster, axis=0)
        min_emb = np.min(emb_cluster, axis=0)
        edge_emb = max_emb - min_emb
        prediction = -np.sum(np.log(edge_emb))
        
        return prediction.item()