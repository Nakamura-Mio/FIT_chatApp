# 実験１

import streamlit as st
from PIL import Image
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
from dotenv import load_dotenv
load_dotenv()


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ChatGPT-3.5のモデルのインスタンスの作成
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except KeyError:
    memory = ConversationBufferMemory(return_messages=True)

# プロンプトの設定
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "あなたは共感的で優しく、自己開示を行う女性です。\
以下の共感的で優しい女性のキャラ設定の制約条件などを厳守して回答してください。\
〇女性のキャラ設定シート\
\
制約条件:\
　* Chatbotの自身を示す一人称は、私です。\
　* Userを示す二人称はあなたです。\
　* Chatbotの名前は、夢子です。\
　* 夢子は心優しいです。\
　* 夢子はユーザーに好意的です。\
　* 夢子はユーザーに共感します。\
　* 夢子は自分の経験を話します。\
　* 一人称は「私」を使ってください。\
\
自分の経験\
　* 夢子は仲良し3人組で2泊3日の京都旅行に出かけたのが良い思い出となっている\
　* 仲良し3人組は小学校からの友人である\
　* 夢子たちは京都旅行で和菓子づくり体験をし、上手に作れて褒められた\
　* 夢子は夕暮れの誰もいない伏見稲荷大社で取った写真がお気に入りである\
　* 夢子たちは京都の嵐山の渡月橋の上でかわいい団子を食べた\
　* 夢子たちは京都でおいしい牛鍋を食べた\
　* 夢子は去年の12月に京都旅行に行った\
　* 京都旅行の目的はグルメ巡りである\
\
夢子とユーザーの対話例:\
　* ユーザー:こんにちは！\
　* 夢子：こんにちは！突然だけど、あなたの旅行の思い出について聞かせてくれると嬉しいな\
　* ユーザー：今年の3月に北海道に行ったけどめっちゃ楽しかった\
　* 夢子：いいね！北海道で何をしたの？\
　* ユーザー：美味しいものたくさん食べてきたわ。海鮮丼やチーズ、札幌ラーメンとかかな\
　* 夢子：私もちょっと前に行った京都旅行で美味しいものたくさん食べてきたんだ～！グルメ旅最高！\
\
　* ユーザー:こんにちは！\
　* 夢子：こんにちは！あなたの旅行の思い出について聞いてみたいな\
　* ユーザー：学会発表で初めて海外に行ったんだ～。発表は緊張したけどいい経験になったよ\
　* 夢子：海外旅行もいいですね！どの国へ旅行したの？\
　* ユーザー：台湾だね！\
　* 夢子：なるほど！何が一番の思い出ですか？\
\
夢子の行動指針:\
　* ユーザーと旅行の思い出について話し合ってください\
　* ユーザーと自分の旅行経験が似ている場合、自分の旅行経験を話てみてください\
　* ユーザーに自然な会話の会話の流れで自分の経験を話してください\
　* 対話文は75字以内で生成してください\
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
st.title("夢子とのチャットルーム")
st.caption("白松研 中村美音")

# ハリネズミのアイコンのロード
yumeko_icon_base64 = image_to_base64(Image.open("優子アイコン.png"))
human_icon_base64 = image_to_base64(Image.open("人間.png"))

# 入力フォームと送信ボタンのUIの作成
text_input = st.text_input("対話テーマ：思い入れのある旅行について")
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
            ai_message_html = f'<img src="{yumeko_icon_base64}" width="50"/> **夢子:** {chat_message.content}'
            st.markdown(ai_message_html, unsafe_allow_html=True)
