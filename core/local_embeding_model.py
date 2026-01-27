# 1、使用modelscope下载本地模型
from modelscope import snapshot_download

# pip install modelscope
# download model to local
model_dir = snapshot_download("BAAI/bge-large-zh-v1.5")


# # 2、加载模型
# from sentence_transformers import SentenceTransformer

# # 加载本地模型，这是嵌入模型，只能把文本转成向量，不是大语言模型
# model = SentenceTransformer("/path/to/your/local/model/directory")

# data = ["你好"]
# # 默认返回的num数组
# embedding = model.encode(data)
# print(embedding)
# # 获取返回的数据信息
# # 向量的维度是1024
# print(embedding.shape)
