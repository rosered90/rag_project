from numpy import where
import chromadb


client = chromadb.Client()

# 支持把向量库存储到本地，实际上sqlite3数据库文件
client = chromadb.PersistentClient(path="./chroma_db")
cll = client.get_or_create_collection(name="doc")

# Add data
cll.add(
    documents=["这是一个测试文档", "这是第二个文档", "这是第三个文档"],
    embeddings=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ids=["1", "2", "3"],
)

# Query data
results = cll.get(ids=["1", "2"], include=["documents", "embeddings"])
results = cll.get(
    ids=["1", "2"],
    include=["documents", "embeddings"],
    where_document={"$contains": "测试"},
)
print(results)


# Delete data
cll.delete(ids=["1"])
print(cll.count())  # Should print 2 after deletion
print(cll.get())  # Should return empty results


# Update data
cll.update(ids=["2"], documents=["这是更新后的第二个文档"], embeddings=[[10, 11, 12]])
print(cll.get(ids=["2"]))  # Should reflect the updated document and embedding
