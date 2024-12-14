import sys
import os

# from py2neo import Graph
from neomodel import StructuredNode, StringProperty, RelationshipTo, config, db

import json
import jieba
import requests

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from novel_qa.settings import BASE_DIR

config.DATABASE_URL = "bolt://neo4j:88888888@localhost:7687"


# 初始化实体识别字典
def init_entity_dict():
    # 从 dict.txt 加载自定义实体词典，每行包括实体及其类型，用制表符 \t 分隔。
    # 将实体名动态添加到 jieba 的分词词典中，便于分词时识别特定领域的实体。
    dict_path = os.path.join(BASE_DIR, "web/data/dict.txt")
    word_dict = [
        i.strip("\n") for i in open(dict_path, "r", encoding="utf-8").readlines()
    ]
    # print(word_dict)

    # 实体名称映射到实体类型
    entity2type = {}
    for word in word_dict:
        tmp_list = word.split("\t")
        # print(tmp_list)
        if len(tmp_list) == 2:
            # 将词典添加到jieba分词的自定义词典中
            jieba.add_word(tmp_list[0])
            entity2type[tmp_list[0]] = tmp_list[1]
    return entity2type


entity2type = init_entity_dict()
# print(entity2type)


# 通过自定义词典获取实体
def get_enyity(input_str):
    word_cut_dict = []
    # 对用户问句进行分词
    for word in jieba.cut(input_str.strip()):
        if word in entity2type:
            print(word)
            word_cut_dict.append(word)
    return list(set(word_cut_dict))


# 提示词模板
PROMPT_TEMPLATE = """已知信息：
{context} 
根据上面提供的三元组信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请你根据你的理解回答用户问题。问题是：{question}"""


PROMPT_TEMPLATE1 = """请你用你的已有知识，简洁和专业的来回答用户的问题。如果无法从中得到答案，请你根据你的理解回答用户问题。问题是：{question}"""


# 获取提示词模板
def get_prompt(question, context):
    # 如果问题和neo4j查询到的上下文都不为空，则拼接出一个提示词，提交给大模型
    if len(question) and len(context) > 0:

        return PROMPT_TEMPLATE.format(context=context, question=question)
    # 如果neo4j查询到的数据为空，则提交问题给大模型
    else:
        return PROMPT_TEMPLATE1.format(question=question)


# 从neo4j查询三元组
def search_entity_from_neo4j(question, entities):
    triplet_list = []
    if not entities:
        return get_prompt(question=question, context="")

    # 查询三元组
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.name IN {entity_names}
    RETURN a.name AS start_node, type(r) AS relation, b.name AS end_node
    """
    params = {"entity_names": entities}
    results, meta = db.cypher_query(query, params)

    for start_node, relation, end_node in results:
        triplet_list.append(f"({start_node},{relation},{end_node})")

    triplet_list = list(set(triplet_list))  # 去重
    return get_prompt(question=question, context="\n".join(triplet_list)[:4096])


def get_answer(input_str):
    input_str = input_str.replace(" ", "").strip()
    # 预测实体
    entitys = get_enyity(input_str)
    print(entitys)
    # 如果识别到实体
    if entitys:
        # 先查询实体相关的三元组，并拼接出提示词
        prompt = search_entity_from_neo4j(input_str, entitys)
        print(prompt)
        # 从大模型获取答案
        answer = get_answer_from_glm3(prompt)
        return answer
    else:
        print("未识别到实体，直接提交问题给大模型")
        prompt = PROMPT_TEMPLATE1.format(question=input_str)
        answer = get_answer_from_glm3(prompt)
        # 返回答案和查询语句
        return answer


def get_answer_from_glm3(prompt):
    url = "http://127.0.0.1:7861/chat/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "query": prompt,
        "history": [],
        "conversation_id": "xxxxxx",
        "history_len": 5,
        "stream": False,
        "model_name": "chatglm3-6b",
        "temperature": 0.7,
        "max_tokens": 0,
        "prompt_name": "default",
    }
    print("发送请求数据:", data)
    response = requests.post(url, headers=headers, data=json.dumps(data))
    body = response.content.decode("utf-8")
    json_data = json.loads(body.replace("data: ", ""))
    print("模型返回数据:", json_data)
    return json_data["text"]

# 设置项目目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
