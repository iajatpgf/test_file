import os
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader, \
    UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(selected_model, openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model=selected_model, openai_api_key=openai_api_key)

    file_content = uploaded_file.read()
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()  # 获取文件扩展名并转为小写

    # 使用临时文件以兼容不同的文件加载器
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # 根据文件扩展名选择合适的加载器
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == '.txt':
        loader = TextLoader(temp_file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(temp_file_path)
    elif file_extension == '.pptx':
        loader = UnstructuredPowerPointLoader(temp_file_path)
    elif file_extension == '.xlsx':
        loader = UnstructuredExcelLoader(temp_file_path)
    else:
        os.unlink(temp_file_path)  # 删除临时文件
        return "不支持的文件类型。"

    docs = loader.load()
    os.unlink(temp_file_path)  # 使用完毕后删除临时文件

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    # 建立文本的向量嵌入并创建快速检索系统
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    # 创建对话检索链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    # 执行问答
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
