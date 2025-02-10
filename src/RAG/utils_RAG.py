import os
import json
import numpy as np
from pyserini.search.lucene import LuceneSearcher
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
Retriever_Name = "bm25"
Corpus = "pubmed"


class Retriever:  # 检索器
    def __init__(self, corpus_name=Corpus, db_dir=current_dir+"/corpus", **kwarg):
        self.corpus_name = corpus_name

        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        if not os.path.exists(self.chunk_dir):
            print("Cloning the {:s} corpus ...".format(self.corpus_name))
            os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus_name,
                                                                                          os.path.join(self.db_dir,
                                                                                                       self.corpus_name)))
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", Retriever_Name)
        if os.path.exists(self.index_dir):
            self.index = LuceneSearcher(self.index_dir)
        else:
            os.system(
                "python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(
                    self.chunk_dir, self.index_dir))
            self.index = LuceneSearcher(self.index_dir)

    def get_relevant_documents(self, question, k=30, **kwarg):
        assert isinstance(question, str)
        question = [question]
        res_ = [[]]
        hits = self.index.search(question[0], k=k)
        res_[0].append(np.array([h.score for h in hits]))
        indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]

        texts = self.idx2txt(indices)
        scores = res_[0][0].tolist()

        return texts, scores

    def idx2txt(self, indices):  # return List of Dict of str
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        '''
        return [json.loads(
            open(os.path.join(self.chunk_dir, i["source"] + ".jsonl")).read().strip().split('\n')[i["index"]]) for i in
            indices]


class RetrievalSystem:

    def __init__(self, db_dir=current_dir+"/corpus"):
        self.retrievers = Retriever(Corpus, db_dir)

    def retrieve(self, question, k=32):
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert isinstance(question, str)
        t, s = self.retrievers.get_relevant_documents(question, k)

        return t, s
