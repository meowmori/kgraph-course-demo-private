import sys
import os

from neomodel import StructuredNode, StringProperty, RelationshipTo, config, db

import json
import jieba
import requests

from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import tkinter as tk
from tkinter import scrolledtext


config.DATABASE_URL = "bolt://neo4j:88888888@localhost:7687"


# 返回实体名称-类别对应的字典
def init_entity_dict():
    # 从 dict.txt 加载自定义实体词典，每行包括实体及其类型，用制表符 \t 分隔
    # 将实体名动态添加到 jieba 的分词词典中，便于分词时识别特定领域的实体
    dict_path = "novel.txt"
    word_dict = [
        i.strip("\n") for i in open(dict_path, "r", encoding="utf-8").readlines()
    ]
    # print(word_dict)

    entity2type_dict = {}
    for word in word_dict:
        tmp_list = word.split("\t")
        # print(tmp_list)
        if len(tmp_list) == 2:
            # 将词典添加到jieba分词的自定义词典中
            jieba.add_word(tmp_list[0])
            entity2type_dict[tmp_list[0]] = tmp_list[1]
    return entity2type_dict


ENTITY2TYPE_DICT = init_entity_dict()
# print(ENTITY2TYPE_DICT)


# 通过自定义词典获取实体
def get_entity(input_str):
    word_cut_dict = []
    # 对用户提问进行分词
    for word in jieba.cut(input_str.strip()):
        if word in ENTITY2TYPE_DICT:
            # print(word)
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
    WHERE a.name IN $entity_names
    RETURN a.name AS start_node, type(r) AS relation, b.name AS end_node
    """
    params = {"entity_names": entities}
    results, meta = db.cypher_query(query, params)

    for start_node, relation, end_node in results:
        triplet_list.append(f"({start_node},{relation},{end_node})")

    triplet_list = list(set(triplet_list))  # 去重
    return get_prompt(question=question, context="\n".join(triplet_list)[:4096])


# 交互式对话类
class InteractiveChat:
    def __init__(self, model):
        self.model = model
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def chat(self, input_str):
        input_str = input_str.replace(" ", "").strip()
        entitys = get_entity(input_str)
        print("识别到的实体: ", entitys)

        if entitys:
            prompt = search_entity_from_neo4j(input_str, entitys)
        else:
            print("未识别到实体，直接提交问题给大模型")
            prompt = PROMPT_TEMPLATE1.format(question=input_str)

        print("提示词:", prompt)
        chain = self.model | StrOutputParser()
        response = chain.invoke(prompt)
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": response})
        print("消息记录: ", self.messages)
        return response

    def clear_history(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]


# 创建前端框架
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("问答系统")
        self.root.geometry("600x600")

        # 主窗口布局
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=5)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.prompt_label = tk.Label(
            self.root, text="请输入你的问题……", font=("Arial", 14), fg="gray"
        )
        self.prompt_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # 对话历史框
        self.chat_history = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state="disabled", height=25, width=70
        )
        self.chat_history.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # 底部区域布局
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=5)
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_columnconfigure(2, weight=1)

        # 输入框
        self.input_box = tk.Entry(self.bottom_frame, font=("Arial", 12))
        self.input_box.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # 按钮
        self.send_button = tk.Button(
            self.bottom_frame,
            text="发送",
            command=self.send_message,
            width=10,
            bg="lightblue",
        )
        self.send_button.grid(row=0, column=1, padx=5, pady=5)

        self.clear_button = tk.Button(
            self.bottom_frame,
            text="清除记录",
            command=self.clear_history,
            width=10,
            bg="lightcoral",
        )
        self.clear_button.grid(row=0, column=2, padx=5, pady=5)

    def send_message(self):
        user_input = self.input_box.get().strip()
        if not user_input:
            return

        # 清空输入框
        self.input_box.delete(0, tk.END)

        # 更新对话历史
        self.update_chat_history("You", user_input)

        # 获取模型回答
        response = interactive_chat.chat(user_input)
        self.update_chat_history("Astrobot", response)

    # 更新对话历史
    def update_chat_history(self, sender, message):
        self.chat_history.config(state="normal")
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state="disabled")

    # 清除对话历史
    def clear_history(self):
        interactive_chat.clear_history()
        self.chat_history.config(state="normal")
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state="disabled")


if __name__ == "__main__":
    # 设置API
    api_key = "sk-53b7201e936b457d8802a9f5ca3605b4"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model_name="qwen-max")
    interactive_chat = InteractiveChat(llm)
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
