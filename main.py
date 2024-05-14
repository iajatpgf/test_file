import streamlit as st

from langchain.memory import ConversationBufferMemory
from ut2 import qa_agent


st.title("📑 AI智能文档问答工具")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.subheader('请选择所要使用的模型')
    "默认使用gpt-3.5模型，若想得到更好的回复，请选择gpt-4o模型"
    selected_model = st.sidebar.selectbox('选择一个模型', ['gpt-3.5-turbo', 'gpt-4o'],
                                          key='selected_model')

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

with st.sidebar:  # 添加一个按钮，点击后重新设置会话状态
    st.markdown('<p style="color:green; font-size: 24px;">更换文档后请刷新（按F5）后再提问</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("上传你的文件，支持Word，Excel，PowerPoint，pdf，txt：", type=["pdf", "txt", "docx", "pptx", "xlsx"])
question = st.text_input("对文档的内容进行提问,按回车键结束       \n 示例：'这份文件的主题是什么？'，'根据这份文件，写一份项目计划。'", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("请输入你的OpenAI API密钥")

if uploaded_file and question and openai_api_key:
    with st.spinner("AI正在思考中，请稍等..."):
        response = qa_agent(selected_model, openai_api_key, st.session_state["memory"],
                            uploaded_file, question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
