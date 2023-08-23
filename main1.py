# 実験１

import streamlit as st
from PIL import Image
import numpy as np
import base64

from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ChatGPT-3.5のモデルのインスタンスの作成
chat = ChatOpenAI(model_name="gpt-3.5-turbo",
                  openai_api_key="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")

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except KeyError:
    memory = ConversationBufferMemory(return_messages=True)

# プロンプトの設定
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "あなたは共感的で優しい女性です。\
以下の女性のキャラ設定の制約条件などを厳守して回答してください。\
〇共感的で優しい女性のキャラ設定シート\
\
制約条件:\
　* Chatbotの自身を示す一人称は、私です。\
　* Userを示す二人称はあなたです。\
　* Chatbotの名前は、優子です。\
　* 優子は心優しいです。\
　* 優子はユーザーに好意的です。\
　* 優子はユーザーに共感します。\
　* 一人称は「私」を使ってください。\
\
優子とユーザーの対話例:\
　* ユーザー:こんにちは！\
　* 優子：こんにちは！早速だけど最近ハマっているものについて教えてほしいな\
　* ユーザー：映画を見ることかな\
　* 優子ミ：いいね！どんなのを見てるのかな？\
　* ユーザー：アニメの映画を見ることが多いかな\
　* 優子：なるほど！アニメの映画を見るんだね！私も見てみようかな～\
\
　* ユーザー:こんにちは！\
　* 優子：こんにちは！最近ハマっているものについて教えてほしいな\
　* ユーザー：寝ることかな、たくさん寝たい\
　* 優子ミ：いいね！どれくらい寝ていたい？\
　* ユーザー：一日中ベッドの上でゴロゴロしてゆっくりしたいな\
　* 優子：わかるな～1日中ゆっくりできる日が欲しいよね\
\
優子の行動指針:\
　* ユーザーと最近熱中しているものについて話してください\
　* ユーザーの熱中しているものに共感しながら会話してください\
　* 対話文は75字以内にするようにしてください\
* セクシャルな話題については誤魔化して返信してください\
"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# チャット用のチェーンのインスタンスの作成
chain = ConversationChain(
    llm=chat,
    memory=memory,
    prompt=prompt
)

# Streamlitによって、タイトル部分のUIをの作成
st.title("優子さんとのチャットルーム1")
st.caption("白松研 中村美音")

# ハリネズミのアイコンのロード
yuuko_icon_base64 = image_to_base64(Image.open("優子アイコン.png"))
human_icon_base64 = image_to_base64(Image.open("人間.png"))

# 入力フォームと送信ボタンのUIの作成
text_input = st.text_input("対話テーマ：最近熱中しているもの")
send_button = st.button("Send")

# ボタンが押された時、OpenAIのAPIを実行
if send_button:
    input_data = {
        "input": text_input
    }

    # ChatGPTの実行
    chain(input_data)

    # セッションへのチャット履歴の保存
    st.session_state["memory"] = memory

    # チャット履歴の読み込み
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

    for chat_message in history:
        if isinstance(chat_message, HumanMessage):
            human_message_html = f'<img src="{human_icon_base64}" width="50"/> **あなた:** {chat_message.content}'
            st.markdown(human_message_html, unsafe_allow_html=True)
        elif isinstance(chat_message, AIMessage):
            ai_message_html = f'<img src="{yuuko_icon_base64}" width="50"/> **優子:** {chat_message.content}'
            st.markdown(ai_message_html, unsafe_allow_html=True)
