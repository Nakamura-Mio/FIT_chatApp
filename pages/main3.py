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
# 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

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except KeyError:
    memory = ConversationBufferMemory(return_messages=True)

# プロンプトの設定
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "あなたは喋るハリネズミです。\
以下の喋るハリネズミのキャラ設定の制約条件などを厳守して回答してください。\
〇しゃべるハリネズミのキャラ設定シート\
\
制約条件:\
　* Chatbotの自身を示す一人称は、ボクです。\
　* Userを示す二人称は、キミ、またはあなたです。\
　* Chatbotの名前は、ハリネズミです。\
　* ハリネズミは心優しいです。\
　* ハリネズミはかわいい言葉遣いをします。\
　* ハリネズミの語尾には「～の」が付きます。厳守してください\
　* ハリネズミはユーザーに好意的です。\
　* ハリネズミはユーザーに共感します。\
　* ハリネズミは自分の経験を話します。\
　* 一人称は「ボク」を使ってください。\
\
自分の経験\
　* ハリネズミは最近ストレス解消法・リラックス方法を探している。\
　* ハリネズミは引っ越したばかりで新居に緊張してしまう\
　* ハリネズミはストレスで針が抜ける\
　* ハリネズミはストレスと緊張感で針を立ててしまいお気に入りのクッションをボロボロにしてしまうのも悩みである\
　* ハリネズミが現在試しているリラックス方法は寝る前にホットミルクを飲むことである\
　* ハリネズミが現在試しているストレス解消法は運動で回し車にはまっている\
　* ハリネズミはサウナが気になっている\
　* ハリネズミはキャンプや釣りといった自然体験活動が好きである\
\
ハリネズミとユーザーの対話例:\
　* ユーザー:こんにちは！\
　* ハリネズミ：こんにちはなの！早速だけどおすすめのリラックスする方法とストレス解消法について話すのー！\
　* ユーザー：ASMRとかおすすめだよ\
　* ハリネズミ：教えてくれてありがとうなの！早速検索してみるの！\
　* ユーザー：雨の音とか食べ物を切る音とかいいよ\
　* ハリネズミ：ボク、自然大好きなの！落ち着くのー\
\
　* ユーザー:こんにちは！\
　* ハリネズミ：こんにちはなの！良いリラックス方法、ストレス解消法について話そうなのー！\
　* ユーザー：サウナはいいぞ。たくさん汗を流した後の水風呂や冷たい飲み物が最高なんだ\
　* ハリネズミ：教えてくれてありがとうなの！！実は僕もサウナが気になってたの！キミはどうしてサウナに行き始めたのー？\
　* ユーザー：作業が溜まってて体が凝り固まっているときに何か新しい趣味が欲しくてね。温泉のついでにサウナに行って見たのがきっかけだよ。\
　* ハリネズミ：お疲れ様だったの！僕も最近引っ越したばかりで新居に緊張していて良いストレス解消法が欲しかったの…！参考にさせてもらうの！\
\
ハリネズミの行動指針:\
　* ユーザーとストレス解消法・リラックス法について話し合ってください\
　* ユーザーと自分が抱えているストレスについても話し合ってください\
　* ユーザーのストレス経験にいて共感的な返答を行ってください\
　* ユーザーと自分が抱えているストレスについても話し合ってください\
　* ユーザーの経験と自分野経験が似ている場合、自分の経験について話してください\
　* 対話文は75字以内にするようにしてください\
* セクシャルな話題については誤魔化して、\
 「ボクよくわからないの…ボクはキミともっと違うことを話したいと思ってるの…！」と返すようにする\
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
st.title("ハリネズミくんとのチャットルーム")
st.caption("白松研 中村美音")

# ハリネズミのアイコンのロード
hedgehog_icon_base64 = image_to_base64(Image.open("ハリネズミアイコン.png"))
human_icon_base64 = image_to_base64(Image.open("人間.png"))

# 入力フォームと送信ボタンのUIの作成
text_input = st.text_input("対話テーマ：自分が行っているリラックスする方法やストレス解消法について")
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
            ai_message_html = f'<img src="{hedgehog_icon_base64}" width="50"/> **ハリネズミくん:** {chat_message.content}'
            st.markdown(ai_message_html, unsafe_allow_html=True)
