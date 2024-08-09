import streamlit as st
import openai
from datetime import datetime, date
import numpy as np
import docx
import re
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import concurrent.futures  # concurrent.futures モジュールをインポート

st.set_page_config(layout="wide")

# OpenAI APIキーを設定
# 背景画像をBase64エンコードする関数
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 背景画像のパス
bg_image_path = "C:/Users/Administrator/Desktop/reazon/bg.jpg"
bg_image_base64 = get_base64_encoded_image(bg_image_path)

# 从CSS文件加载样式
def load_css(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        css = file.read()
    return css.replace("{bg_image_base64}", bg_image_base64)

# 应用CSS样式
css_styles = load_css("C:/Users/Administrator/Desktop/reazon/styles.css")
st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

# 从文本文件加载事件数据
def load_event_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return json.loads(data)

# 从文本文件加载酒店数据
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 加载事件数据
events_data = load_event_data("C:/Users/Administrator/Desktop/reazon/event.txt")

def background_func(background_info):
    return (f"あなたは以下の背景情報を持つ旅行ガイドです。{background_info}...")

# Word文書を読み込む関数
def read_paragraphs(paragraphs):
    return [paragraph.text for paragraph in paragraphs]

def read_word_document(file_path):
    doc = docx.Document(file_path)
    paragraphs = doc.paragraphs

    with ThreadPoolExecutor() as executor:
        results = executor.map(read_paragraphs, [paragraphs])

    full_text = [text for result in results for text in result]
    return '\n'.join(full_text)

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return np.array(response['data'][0]['embedding'])

def find_most_relevant_section(document_text, query, top_n=1):
    paragraphs = document_text.split('\n')
    paragraph_embeddings = [get_embedding(p) for p in paragraphs]
    query_embedding = get_embedding(query)
    similarities = [np.dot(query_embedding, p_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(p_emb)) for p_emb in paragraph_embeddings]
    most_relevant_idx = np.argsort(similarities)[-top_n:]
    return [paragraphs[i] for i in most_relevant_idx]

def generate_prompt(location, how, time1, time2):
    user_prompt = f"私に旅行プランを立ててほしい。行先などの情報は以下の通りです。私は{location}に旅行に行きたい。到着日は{time1}で、帰還日は{time2}です。"
    if how == "レンタカー":
        user_prompt += "私は地元でレンタカーを借りたいので、レンタカー比較サービスサイトを出してください。旅行プランを立ててくれる時は駐車場も注意して提案してください。"
    if how == "マイカー":
        user_prompt += "旅行プランを立ててくれる時は駐車場も注意して提案してください。"
    if how == "公共交通":
        user_prompt += "旅行プランを立ててくれる時は公共交通にかかる時間も注意して提案してください。"
    return user_prompt + "時間帯ごとに訪れる場所と各場所の簡単な説明も含めてください。"

def extract_event_dates(event_period_str):
    event_period_str = re.sub(r'（.*?）', '', event_period_str)
    date_ranges = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日)～(\d{1,2}月\d{1,2}日)', event_period_str)
    if not date_ranges:
        single_dates = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日)', event_period_str)
        if single_dates:
            single_date = datetime.strptime(single_dates[0], "%Y年%m月%d日").date()
            return [(single_date, single_date)]
    date_pairs = []
    for start_date_str, end_date_str in date_ranges:
        start_date = datetime.strptime(start_date_str.strip(), "%Y年%m月%d日").date()
        end_date = datetime.strptime(end_date_str.strip(), "%m月%d日").date()
        end_date = end_date.replace(year=start_date.year)
        date_pairs.append((start_date, end_date))
    return date_pairs

def generate_event_recommendation(user_input, events_data):
    prompt = (
        "以下のイベントリストに基づいて、ユーザーのリクエストに最も合ったイベントを推奨し、その理由を説明してください。\n\n"
        "イベントリスト:\n"
    )
    for event in events_data:
        prompt += (
            f"イベント名: {event['イベント名']}\n"
            f"開催地: {event['開催地']}\n"
            f"開催期間: {event['開催期間']}\n"
            f"料金: {event['料金']}\n"
            f"説明: {event['説明']}\n\n"
        )
    prompt += f"ユーザーのリクエスト: {user_input}\n"
    prompt += "最も適したイベント、その理由、および該当イベントの詳細情報を提供してください。"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "あなたはイベントプランナーです。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        stream=True
    )

    full_text = ""
    message_placeholder = st.empty()
    for chunk in response:
        if "choices" in chunk:
            chunk_message = chunk["choices"][0]["delta"].get("content", "")
            full_text += chunk_message
            message_placeholder.write(full_text)

# ホテル情報を解析する関数
def parse_hotel_information(hotel_text):
    hotels = []
    lines = hotel_text.split('\n')

    for line in lines[1:]:
        if line.strip():
            try:
                parts = line.split(',')
                if len(parts) != 8:
                    print(f"解析できない行: {line}")
                    continue

                hotel = {
                    'name': parts[0].strip(),
                    'location': parts[1].strip(),
                    'distance': parts[2].strip(),
                    'score': float(parts[3].strip()),
                    'location_score': float(parts[4].strip()) if parts[4].strip() != 'なし' else None,
                    'room_description': parts[5].strip(),
                    'original_price': int(parts[6].replace('￥', '').replace(',', '').strip()),
                    'current_price': int(parts[7].replace('￥', '').replace(',', '').strip())
                }
                hotels.append(hotel)
            except (ValueError, IndexError) as e:
                print(f"ホテル情報の解析中にエラーが発生しました: {e}")
                continue
    return hotels

# ホテルを推薦する関数（GPTを使用しながら逐步输出）
# ホテルを推薦する関数（GPTを使用しながら逐步输出）
def recommend_hotels_by_price_with_gpt_stream(hotels, budget, top_n=5):
    # 予算以内のホテルをフィルタリング
    filtered_hotels = [hotel for hotel in hotels if hotel['current_price'] <= budget]

    # 価格でソート（低い順）
    sorted_hotels = sorted(filtered_hotels, key=lambda x: x['current_price'])

    # GPTに推薦結果を生成
    if sorted_hotels:
        hotels_info = "\n".join(
            [f"ホテル名: {hotel['name']}, 位置: {hotel['location']}, 距離: {hotel['distance']}, "
             f"評価: {hotel['score']}/10, 現在の価格: {hotel['current_price']}円, 部屋説明: {hotel['room_description']}"
             for hotel in sorted_hotels[:top_n]]
        )
        prompt = (
            f"ユーザーの予算は{budget}円です。以下の予算内で利用可能なホテルの情報に基づいて、"
            "ユーザーに最適なホテルを推奨してください。\n\n"
            f"{hotels_info}\n"
        )
    else:
        prompt = f"ユーザーの予算は{budget}円ですが、予算内に該当するホテルがありません。"

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは旅行プランナーです。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        stream=True
    )

    full_text = ""
    message_placeholder = st.empty()
    for chunk in response:
        if "choices" in chunk:
            chunk_message = chunk["choices"][0]["delta"].get("content", "")
            full_text += chunk_message
            message_placeholder.write(full_text)


def main():
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>紅葉の旅</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        div.row-widget.stRadio > div {flex-direction:row;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    options = ['そこに関してもっと知りたい', '観光アドバイスがほしい', '期間限定 イベント情報', 'ホテル選択']
    selected_option = st.radio('', options, index=0)

    if selected_option == 'そこに関してもっと知りたい':
        question = st.text_input('聞きたい観光地:')
        is_generate_clicked = st.button("検索を開始")
        if is_generate_clicked and question:
            document_text = read_word_document("./eth.docx")
            background_info = find_most_relevant_section(document_text, question)
            system_prompt1 = background_func(background_info)
            system_text = system_prompt1
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": question}
                ],
                max_tokens=2000,
                temperature=0,
                stream=True
            )
            partial_words = ""
            answer = st.empty()
            for chunk in response:
                if 'choices' in chunk:
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    partial_words += chunk_message
                    answer.write(partial_words)

    elif selected_option == '観光アドバイスがほしい':
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox('旅行の行先を選択してください:', ("京都"))
        with col2:
            how = st.selectbox('移動手段を選択してください:', ("マイカー", "レンタカー", "公共交通"))
        col3, col4 = st.columns(2)
        with col3:
            time1 = st.date_input('到着日を選択してください:', datetime.now())
        with col4:
            time2 = st.date_input('帰還日を選択してください:', datetime.now())
        st.text_input('具体的なニーズをここで述べてください:')
        is_generate_clicked = st.button("旅行プランを生成")
        if is_generate_clicked:
            question = generate_prompt(location, how, time1, time2)
            system_text = "あなたは自治体の有能な旅行ガイダンスAIアシスタントで、ミッションはユーザーのニーズに応じて、旅行プランを立てることです。"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": question}
                ],
                max_tokens=2000,
                temperature=0,
                stream=True
            )
            partial_words = ""
            answer = st.empty()
            for chunk in response:
                if 'choices' in chunk:
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    partial_words += chunk_message
                    answer.write(partial_words)

    elif selected_option == '期間限定 イベント情報':
        mode = st.radio("イベント検索方法を選択してください",
                        ("イベント名で検索", "日付と予算で検索", "自然言語で検索"))

        if mode == "イベント名で検索":
            event_names = [event["イベント名"] for event in events_data]
            selected_event_name = st.selectbox("表示したいイベントを選んでください", event_names)

            if selected_event_name:
                selected_event = next(event for event in events_data if event["イベント名"] == selected_event_name)
                st.subheader(f"イベント名称: {selected_event['イベント名']}")
                st.write(f"開催地: {selected_event['開催地']}")
                st.write(f"開催期間: {selected_event['開催期間']}")
                st.write(f"料金: {selected_event['料金']}")
                st.write(f"説明: {selected_event['説明']}")

        elif mode == "日付と予算で検索":
            selected_date = st.date_input("来訪予定の日付を選択してください", value=date.today())
            budget = st.number_input("予算を入力してください（円）", min_value=0, value=1000)

            matching_events = []
            for event in events_data:
                date_ranges = extract_event_dates(event["開催期間"])
                for event_start_date, event_end_date in date_ranges:
                    if event_start_date <= selected_date <= event_end_date and event["料金数値"] <= budget:
                        matching_events.append(event)
                        break

            if matching_events:
                st.subheader("検索結果:")
                for event in matching_events:
                    st.write(f"イベント名称: {event['イベント名']}")
                    st.write(f"開催地: {event['開催地']}")
                    st.write(f"開催期間: {event['開催期間']}")
                    st.write(f"料金: {event['料金']}")
                    st.write(f"説明: {event['説明']}")
                    st.write("---")
            else:
                st.write("条件に合うイベントが見つかりませんでした。")

        elif mode == "自然言語で検索":
            user_input = st.text_input("希望のイベントについて教えてください:")
            if st.button("おすすめを生成"):
                if user_input:
                    generate_event_recommendation(user_input, events_data)
                else:
                    st.write("入力をしてください。")

    elif selected_option == 'ホテル選択':
        hotel_text = read_text_file("C:/Users/Administrator/Desktop/reazon/hotel.txt")
        hotels = parse_hotel_information(hotel_text)

        user_budget = st.number_input("ご予算を入力してください（円）", min_value=0, value=10000)
        if st.button("ホテルを推薦"):
            recommend_hotels_by_price_with_gpt_stream(hotels, user_budget)

if __name__ == "__main__":
    main()
