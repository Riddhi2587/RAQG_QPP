import pyterrier as pt
import pandas as pd
import numpy as np
from pyterrier.model import add_ranks

class KLDivReranker:

    def __init__(self, sampled_res, stats, feedback_weight_dict):
        self.sampled_res = sampled_res
        self.stats = stats
        self.feedback_weight_dict = feedback_weight_dict
    
    def rerank_docs(self):
        kl_div_score_dict = {}

        for docid in self.sampled_res.docid:
            
            tempt_kl_score = 0
            doc_stats = self.stats['in_docs_tf_per_doc'][docid]
            # print('[debug inf error]', doc_stats)

            for t, _ in self.stats['in_docs_tf'].items():
                
                tf = doc_stats.get(t, 0)
                if(tf==0):
                    continue
                    
                normalised_tf = tf / doc_stats['sum_tf']
                tempt_kl_score += self.feedback_weight_dict[t] * np.log(self.feedback_weight_dict[t] / normalised_tf)
                
            kl_div_score_dict.update({docid: (1 - np.exp(tempt_kl_score))})

        reranked_sampled_res = self.sampled_res.copy()  # here the score is just a fake score added in shuffling, so just overriding it
        reranked_sampled_res["score"] = reranked_sampled_res.docid.apply(lambda x: kl_div_score_dict[x])
        reranked_sampled_res = reranked_sampled_res.drop(columns = ["rank"])
        reranked_sampled_res = add_ranks(reranked_sampled_res)

        return reranked_sampled_res

class RelevanceModelBase:

    def __init__(self, index, query_df, sampled_res, k:int):
        self.index = index
        self.query_text = query_df['query'].values[0]
        self.sampled_res = sampled_res
        self.statistics = {}
        self.feedback_weights = {}
        self.k = k
        self.MIX_COEFF = 0.9
        self.set_stopwords()

    def get_corpus_document_frequency(self, _token):
        meta = self.index.meta_index()
        lexicon = self.index.lexicon()
        total_tokens_freq = self.index.collection_statistics().numberOfDocuments

        try:
            if(_token in self.stopwords):
                d_freq = total_tokens_freq
            else:
                d_freq = lexicon[_token].getDocumentFrequency()
        except Exception as e:
            print(f'error: {_token} doesn\'t exist in the vocabulary, set d_freq to 1.')
            d_freq = 1

        return d_freq

    def get_query_lm(self):
        stemmer = pt.TerrierStemmer.porter
    
        tf_dict = {}
        df_dict = {}
        for token in self.query_text.split(' '):
            t = stemmer.stem(token)
            tf_dict[t] = tf_dict.get(t, 0) + 1
            df_dict.update({t: self.get_corpus_document_frequency(t)})
                
        return {'in_query_tf': tf_dict, 'corpus_df_for_q': df_dict}, sum(tf_dict.values())

    def get_corpus_lm(self):

        di = self.index.direct_index()
        doi = self.index.document_index()
        lexicon = self.index.lexicon()
        
        tf_dict = {}
        tf_dict_per_doc = {}
        corpus_df_dict = {}
        local_df_dict = {}
        
        for docid in self.sampled_res.docid:
            # print(docid)
            tf_dict_per_doc.update({docid: {}})
            
            for posting in di.getPostings(doi.getDocumentEntry(docid)):
                termid = posting.getId()
                lee = lexicon.getLexiconEntry(termid)
                tf_dict[lee.getKey()] = tf_dict.get(lee.getKey(), 0) + posting.getFrequency()
                tf_dict_per_doc[docid][lee.getKey()] = tf_dict_per_doc[docid].get(lee.getKey(), 0) + posting.getFrequency()
                corpus_df_dict.update({lee.getKey(): self.get_corpus_document_frequency(lee.getKey())})
                local_df_dict[lee.getKey()] = local_df_dict.get(lee.getKey(), 0) + 1

            tf_dict_per_doc[docid]['sum_tf'] = sum(tf_dict_per_doc[docid].values())
            
        return {'in_docs_tf': tf_dict, 'in_docs_tf_per_doc': tf_dict_per_doc, 'corpus_df': corpus_df_dict, 'local_df': local_df_dict}

    def mix_tf_idf(self, statistics, token):
        
        # print(token, '[tf_ratio=]', statistics['in_docs_tf'][token]/statistics['sum_tf'], '[df_ratio=]', statistics['local_df'][token]/statistics['sum_df'])
        tf_ratio = statistics['in_docs_tf'][token]/statistics['sum_tf']
        df_ratio = statistics['local_df'][token]/statistics['sum_df']
        return self.MIX_COEFF * tf_ratio + (1-self.MIX_COEFF)*df_ratio

    def compute_feedback_weights(self):
        statistics = self.get_query_lm()[0]
        statistics.update(self.get_corpus_lm())
        statistics['sum_tf'] = sum(statistics['in_docs_tf'].values())
        statistics['sum_df'] = sum(statistics['local_df'].values())
        self.statistics = statistics

        total_p_q = 0
        for qt, qtf in statistics['in_query_tf'].items():
            qt_freq_in_docs = statistics['in_docs_tf'].get(qt, -1)
            if(qt_freq_in_docs == -1):
                continue
            else:
                p_q = qt_freq_in_docs/statistics['sum_tf']
                total_p_q += np.log(1+p_q)
        
        for t in statistics['local_df'].keys():
            p_w = self.mix_tf_idf(statistics, t)
            wt = p_w * np.exp(total_p_q-1)
            self.feedback_weights.update({t: wt})

    def rerank_docs(self):
        self.reranker = KLDivReranker(self.sampled_res, self.statistics, self.feedback_weights)
        return self.reranker.rerank_docs()

    def set_stopwords(self, file_path="./stopword-list.txt"):

        self.stopwords = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # strip whitespace & skip empty/comment lines
                word = line.strip()
                if word:
                    self.stopwords.append(word)
        
        # print(f"Loaded {len(self.stopwords)} stopwords.")

class RelevanceModelConditional(RelevanceModelBase):

    def mix_tf_idf(self, statistics, doc_tf_dict, token):

        tf_ratio = doc_tf_dict[token]/doc_tf_dict['sum_tf']
        df_ratio = statistics['local_df'][token]/statistics['sum_df']
        return self.MIX_COEFF * tf_ratio + (1-self.MIX_COEFF)*df_ratio

    def compute_feedback_weights(self):
        statistics = self.get_query_lm()[0]
        statistics.update(self.get_corpus_lm())
        statistics['sum_tf'] = sum(statistics['in_docs_tf'].values())
        statistics['sum_df'] = sum(statistics['local_df'].values())
        self.statistics = statistics

        # print(statistics['in_docs_tf_per_doc'])

        for docid, orig_score in self.sampled_res[['docid', 'orig_score']].values:

            for t in statistics['in_docs_tf_per_doc'][docid].keys():   
                if(t == 'sum_tf'):
                    continue   # the last is the sum_tf key
                p_w = self.mix_tf_idf(statistics, statistics['in_docs_tf_per_doc'][docid], t)
                wt = p_w * (orig_score/self.sampled_res.orig_score.sum())

                self.feedback_weights[t] = self.feedback_weights.get(t, 0) + wt
            