# One-time build of the MS MARCO passage corpus index needed for RBO reranking
# in retrieve_qvs.py (--doc_index_path points here once built).
#
# Usage:
#   python doc_indices/build_msmarco_passage_index.py --out_path ./doc_indices/msmarco-passage.terrier

import argparse
import pyterrier as pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default="./doc_indices/msmarco-passage.terrier")
    args = parser.parse_args()

    dataset = pt.get_dataset("irds:msmarco-passage")
    indexer = pt.IterDictIndexer(args.out_path, meta={"docno": 20, "text": 4096})
    indexer.index(dataset.get_corpus_iter(), fields=["text"])
    print(f"[progress] Index built at {args.out_path}")
