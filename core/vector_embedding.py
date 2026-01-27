from common import rag_client


def get_embeding(text):
    """
    get vector embedding for given text
    """
    data = rag_client.embeddings.create(input=text, model="text-embedding-v1")
    print(data)
    return [i.embedding for i in data.data]


test_qurery = "大模型是怎么有记忆呢，为什么我们和它说话它能记住之前的内容？"

# v1向量的维度是：1536
# v2向量的维度是：1024
# V3向量的维度是：1536
# 具体用了多少维度的向量，这个和用不同的模型有关系
vec = get_embeding(test_qurery)

# return length of the embedding vector
print(len(vec[0]))
