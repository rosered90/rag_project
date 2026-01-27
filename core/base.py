import chromadb
from dotenv import load_dotenv
from vector_embedding import rag_client
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_pages


class PDFConnector:

    # 重叠分割文档
    @staticmethod
    def sliding_window_chunks(text, chunk_size=500, stride=50):
        """
        Split text into chunks using a sliding window approach.
        """
        return [text[i : i + chunk_size] for i in range(0, len(text), stride)]

    # 读取pdf
    def extract_text_from_pdf(self, file_name, page_numbers=None):
        """从PDF文件中(按照指定页码)提取文本内容"""
        paragraphs = []
        full_text = ""

        for i, page_layout in enumerate(extract_pages(file_name)):
            if page_numbers is not None and i not in page_numbers:
                continue
            for element in page_layout:
                # 检查element是否为文本
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text().replace("\n", "").replace(" ", "")

        if full_text:
            text_chunks = self.sliding_window_chunks(
                full_text, chunk_size=500, stride=50
            )
            for chunk in text_chunks:
                paragraphs.append(chunk)
        return paragraphs


class RAGBot():
    def __init__(self, vector_db, n_res=4):
        self.vector_db = vector_db
        self.n_res = n_res

    def get_completion(self, prompt):
        response = rag_client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False,
        )
        return response.choices[0].message.content

    def chat(self, question):
        # 在向量数据库中搜索相关内容
        results = self.vector_db.search(question, n_results=self.n_res)
        context = "\n".join(results["documents"][0])
        print("Context for the question:", context)

        prompt_tmplate = """
            你是一个问答机器人
            你的任务是根据下述给定的已知信息回答用户问题
            确保你的额回复完全依据下述已知信息。不要编造答案。
            如果需下述已知信息不足以回答用户的问题，请直接回复“我无法回答你的问题”。
            已知信息：
            {context}

            用户问题：{question}
            请用中文回答用户问题
        """
        prompt_tmplate = prompt_tmplate.replace("{context}", context).replace("{question}", question)
        print("Final Prompt:", prompt_tmplate)
        response = self.get_completion(prompt_tmplate)
        print("Response:", response)
        return response


class VectorDBConnector:
    def __init__(self):
        self.db = chromadb.Client()
        self.collection = self.db.get_or_create_collection(name="doc")

    def get_embeddings(self, texts, model="text-embedding-v1"):
        """把数据转换成向量"""
        results = rag_client.embeddings.create(input=texts, model=model).data
        return [item.embedding for item in results]
    
    def create_documents(self, instructions):
        """
        把指令数据添加到向量数据库中，docuemnt是指令内容
        
        :param self: 说明
        :param instructions: 说明
        """
        embeddings = self.get_embeddings(instructions)
        self.collection.add(
            embeddings=embeddings,
            documents=instructions,
            ids=[str(i) for i in range(len(instructions))],
        )

    def add_documents(self, instructions, outputs):
        embeddings = self.get_embeddings(instructions)
        self.collection.add(
            embeddings=embeddings,
            documents=outputs,
            ids=[str(i) for i in range(len(outputs))],
        )

    def search(self, query, n_results=4):
        """
        把我们查询的问题向量化，在chroma中进行相似度搜索
        """
        results = self.collection.query(
            query_embeddings=self.get_embeddings([query]), n_results=n_results
        )
        return results


if __name__ == "__main__":
    load_dotenv()
    vector_db = VectorDBConnector()
    docs = [
        {
            "instruction": "这是一个测试文档",
            "output": "这是你的初入的公司，在这公司你遇到了一群可爱的同事",
        },
        {
            "instruction": "这是第二个文档",
            "output": "这是第二个公司，虽然是一个小型的公司，但是你感受了另外一种可能",
        },
        {
            "instruction": "这是第三个文档",
            "output": "这是现在的公司，公司氛围很好，大家都很努力",
        },
    ]
    instruction = [doc["instruction"] for doc in docs]
    # outputs = [doc["output"] for doc in docs]
    # # 将问题和回答添加到向量数据库中
    # vector_db.add_documents(instruction, outputs)

    # user_query = "第二个文档？"
    # results = vector_db.search(user_query)
    # print(results)

    # PDF处理测试
    # pdf_connector = PDFConnector()
    # paragraphs = pdf_connector.extract_text_from_pdf(
    #     "./static/test1.pdf", page_numbers=[0, 1, 2, 3, 4]
    # )
    # print(paragraphs)

    # Bot
    
    vector_db.create_documents(instruction)
    rag_bot = RAGBot(vector_db=vector_db, n_res=4)

    # 创建机器人对象
    rag_bot = RAGBot(vector_db=vector_db, n_res=4)
    rag_bot.chat("第二个文档？")

