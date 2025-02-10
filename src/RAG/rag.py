# Description: This file contains the implementation of the MedRAG,
# which is used to interact with the RAG model.
# written by:Jacob Zhang
# Date: 2025-1-22
# ver: 0.2

# 任务：仅返回 Retrieved 到的文本

# 下一步：1.统一api  2.将pubmed改为百度云的数据集
import tiktoken
from .utils_RAG import RetrievalSystem
from .template import *
from src.ClinicInteract.cofig import Config
import re
from openai import OpenAI
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
llm_name = "OpenAI/gpt-3.5-turbo-16k"
rag = True
corpus_name = "PubMed"
db_dir = current_dir + "/corpus"


def _initialize_model():
    global tokenizer
    global context_length
    global max_length
    global model

    if llm_name.split('/')[0].lower() == "openai":
        model = llm_name.split('/')[-1]
        # 斜杠后面的那个。[-1]其实和【1】是一样的，只不过是从后往前数的。
        # self.model参数是具体的模型名字，比如gpt-3.5-turbo-16k
        if "gpt-3.5" in model or "gpt-35" in model:
            max_length = 16384
            context_length = 15000
        elif "gpt-4" in model:
            max_length = 32768
            context_length = 30000
        tokenizer = tiktoken.get_encoding("cl100k_base")


def init():
    global retrieval_system
    global templates
    retrieval_system = RetrievalSystem(db_dir)

    templates = {"cot_system": general_cot_system,
                 "cot_prompt": general_cot,
                 "rag_system": general_medrag_system,
                 "rag_prompt": general_medrag
                 }
    _initialize_model()


def answer(question, options=None, k=32, return_rag_result_only=True):
    '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from   #如果选项数是固定的，那么也可以使用list
        后面可能要改，因为回答不再是选项而是一个答案。
        k (int): number of snippets to retrieve #检索片段数，
        k和rrf的区别：k是最终返回给大模型的片段数，rrf是从融合数据库中检索的片段数。rrf大于等于k，
        有一个打分模块对这些片段进行打分，然后选取分数最高的k个片段返回给大模型。，这就是rrf
        rrf_k主要用于检索融合。检索融合可能要删除，因为MedCPT的扩展性一般
        '''
    init()
    global tokenizer

    if options is not None:
        options = '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
    else:
        options = ''

    # retrieve relevant snippets 检索相关片段
    answers = []
    if rag:
        retrieved_snippets, scores = retrieval_system.retrieve(question, k=k)
        # 因为这是个检索模型，所以它的retrieve方法返回的是检索到的片段和它们的分数。如果不需要模型融合，那么rrf_k就是k
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"],
                                                                retrieved_snippets[idx]["content"]) for idx in
                    range(len(retrieved_snippets))]
        if len(contexts) == 0:
            contexts = [""]
        # 根据 self.llm_name 的不同，使用 self.tokenizer 对 contexts 列表进行编码和截断处理，确保上下文长度不超过
        # self.context_length。
        contexts = [tokenizer.decode(tokenizer.encode("\n".join(contexts))[:context_length])]

        if return_rag_result_only:
            return contexts, scores

        for context in contexts:
            prompt_medrag = templates["rag_prompt"].render(context=context, question=question,
                                                           options=options)
            messages = [
                {"role": "system", "content": templates["rag_system"]},
                {"role": "user", "content": prompt_medrag}
            ]
            print(messages)
            ans = generate(messages)
            answers.append(re.sub("\s+", " ", ans))

    else:
        retrieved_snippets = []
        scores = []

        # generate answers
        prompt_cot = templates["cot_prompt"].render(question=question, options=options)
        messages = [
            {"role": "system", "content": templates["cot_system"]},
            {"role": "user", "content": prompt_cot}
        ]
        ans = generate(messages)
        answers.append(re.sub("\s+", " ", ans))

    return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores


def generate(messages):
    """
        generate response given messages
        """
    if "openai" in llm_name.lower():
        client = OpenAI(
            api_key=Config.openai_api_key,
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        ans = response.choices[0].message.content
    return ans


if __name__ == "__main__":
    # Test
    question = """A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. 
    During the case, the resident inadvertently cuts a flexor tendon. 
    The tendon is repaired without complication. 
    The attending tells the resident that the patient will do fine, 
    and there is no need to report this minor complication that will not harm the patient, 
    as he does not want to make the patient worry unnecessarily. 
    He tells the resident to leave this complication out of the operative report. 
    Which of the following is the correct next action for the resident to take?"""
    options = {
        "A": "Disclose the error to the patient but leave it out of the operative report",
        "B": "Disclose the error to the patient and put it in the operative report",
        "C": "Tell the attending that he cannot fail to disclose this mistake",
        "D": "Report the physician to the ethics committee",
        "E": "Refuse to dictate the operative report"
    }

    init()

    answer = answer(question=question, options=None, return_rag_result_only=True)
    print(answer[0])
    print(answer[1])
