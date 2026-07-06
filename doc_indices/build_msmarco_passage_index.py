# Obtain the MS MARCO passage corpus index needed for RBO reranking in retrieve_qvs.py
# (--doc_index_path points here once built/downloaded).
#
# By default, tries to download PyTerrier's prebuilt "terrier_stemmed" index from the
# Terrier Data Repository (data.terrier.org) - fast, no local indexing needed. Falls back
# to building the index from scratch via ir_datasets if the prebuilt download fails or
# --no_prebuilt is passed.
#
# Usage:
#   python doc_indices/build_msmarco_passage_index.py --out_path ./doc_indices/msmarco-passage.terrier

import argparse
import pyterrier as pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default="./doc_indices/msmarco-passage.terrier",
                         help="Where to build the index from scratch if the prebuilt index isn't used.")
    parser.add_argument("--variant", type=str, default="terrier_stemmed",
                         help="Prebuilt index variant to download from the Terrier Data Repository.")
    parser.add_argument("--no_prebuilt", action="store_true",
                         help="Skip the prebuilt-index download and always build from scratch.")
    args = parser.parse_args()

    index_path = None
    if not args.no_prebuilt:
        try:
            dataset = pt.get_dataset("msmarco_passage")
            index_path = dataset.get_index(args.variant)
            print(f"[progress] Downloaded prebuilt '{args.variant}' index to {index_path}")
        except Exception as e:
            print(f"[progress] Prebuilt index unavailable ({e}); building from scratch instead.")

    if index_path is None:
        dataset = pt.get_dataset("irds:msmarco-passage")
        indexer = pt.IterDictIndexer(args.out_path, meta={"docno": 20, "text": 4096})
        indexer.index(dataset.get_corpus_iter(), fields=["text"])
        index_path = args.out_path
        print(f"[progress] Index built at {index_path}")

    print(f"RESULT_PATH:{index_path}")
