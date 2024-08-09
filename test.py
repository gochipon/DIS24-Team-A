import streamlit as st
import openai
import os
import docx
import numpy as np
from datetime import datetime, date
import re
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static
from fpdf import FPDF

openai.api_key = ""
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Travel plan Only for you!', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

def extract_time_and_location(response):
    time_pattern = r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b'
    location_pattern = r'(京都駅|清水寺|東本願寺|西本願寺|京都タワー|京都ポルタ|祇園|金閣寺|二条城)'

    times = re.findall(time_pattern, response)
    locations = re.findall(location_pattern, response)

    return times, locations

def get_coordinates(location):
    geolocator = Nominatim(user_agent="travel_planner")
    location_obj = geolocator.geocode(location)
    if location_obj:
        return location_obj.latitude, location_obj.longitude
    else:
        return None, None

def display_map(locations):
    if locations:
        lat, lon = get_coordinates(locations[0])
        if lat and lon:
            m = folium.Map(location=[lat, lon], zoom_start=13)
            for loc in locations:
                lat, lon = get_coordinates(loc)
                if lat and lon:
                    folium.Marker([lat, lon], tooltip=loc).add_to(m)
            st.session_state.map = m  # 地図オブジェクトをセッションステートに保存
        else:
            st.error("None.")
    else:
        st.warning("None.")

events_data =  {"京都": [
    {
        "イベント名": "第49回京の夏の旅　長楽館「御成の間」",
        "開催地": "長楽館「御成の間」",
        "開催期間": "2024年7月12日（金）～9月30日（月）",
        "料金": "大人800円／小学生400円",
        "料金数値": 800,
        "説明": ("長楽館は明治四十二年に実業家村井吉兵衛氏により迎賓館として建築されました。"
                "外観はルネッサンス風の洋館で、内部はロココ、ネオ・クラシック、アールヌーボーといった"
                "様々な建築様式が取り入れられているのが特徴で、京都市有形文化財に指定されています。"
                "現在ではカフェとホテルからなり、円山公園や東山を眺めながら優雅なひとときを楽しむことが出来ます。"
                "特別公開の「御成の間」はその最上階である3階に位置し、バカラ社製のシャンデリアに折上格天井、"
                "波に千鳥を描いた金地の襖絵などが和洋折衷の特別な空間です。"
                "表千家の「残月亭」写しと伝わる茶室「長楽庵」など、普段は非公開部分を見ることができます。")
    },
    {
        "イベント名": "第73回 亀岡平和祭保津川市民花火大会",
        "開催地": "亀岡市保津橋上流",
        "開催期間": "2024年8月11日",
        "料金": "詳細は公式HPをご確認ください",
        "料金数値": 0,  # このイベントは料金が不明なので0にしている
        "説明": ("10,000発の花火打ち上げに加え、西日本最大級の2尺玉を内閣総理大臣賞受賞経験もある"
                "（株）マルゴーを中心とした特別チームが打ち上げます。"
                "2尺玉は、開いた花火の直径が500ｍにもなる特大の花火で、夜空を覆いつくす大きさと魂に響く轟音は圧巻です。")
    },
    {
        "イベント名": "嵐電 妖怪電車",
        "開催地": "嵐山本線（四条大宮～嵐山）",
        "開催期間": "2024年8月14日(水)〜08月15日(木)、08月17日(土)〜08月18日(日)",
        "料金": "詳細は公式HPをご確認ください",
        "料金数値": 0,  # このイベントは料金が不明なので0にしている
        "説明": ("「嵐電 妖怪電車」は、お化け屋敷風に装飾され数体の妖怪が乗り込んだ嵐電に、"
                "お客様自身も妖怪に扮装するなどしてご乗車いただき、車内一体で楽しむ参加型のイベントです。")
    },
    {
        "イベント名": "東映太秦映画村 冷やし盆踊り",
        "開催地": "東映太秦映画村",
        "開催期間": "2024年07月13日(土)〜08月18日(日)",
        "料金": "大人 2400円",
        "料金数値": 2400,
        "説明": "東映太秦映画村では、ヨーロッパ企画・酒井善史 脚本・演出による江戸の水かけショー「納涼！冷やし盆踊り」を開催します。"
    },
    {
        "イベント名": "アジアンマーケット",
        "開催地": "ジェイアール京都伊勢丹 10階 催物場",
        "開催期間": "2024年08月13日(火)〜08月18日(日)",
        "料金": "入場料無料",
        "料金数値": 0,  # 無料イベントなので0にしている
        "説明": "ジェイアール京都伊勢丹では、アジア各国の食と雑貨が集まる「アジアンマーケット」を開催します。"
    },
    {
        "イベント名": "NAKED 夏まつり2024 世界遺産・二条城",
        "開催地": "二条城",
        "開催期間": "2024年07月26日(金)〜08月25日(日)",
        "料金": "中学生以上1,800円～",
        "料金数値": 1800,
        "説明": "世界遺産 二条城では、夜間に夏まつり気分を満喫いただくため、「NAKED 夏まつり2024 世界遺産・二条城」を開催します。"
    }
]
}
# GPTを使用してイベントを推薦する関数（逐步显示）
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたはイベントプランナーです。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        stream=True  # ストリーミングモードを有効にする
    )

    # ストリーミングレスポンスの逐次処理
    full_text = ""
    message_placeholder = st.empty()  # プレースホルダーを作成
    for chunk in response:
        if "choices" in chunk:
            chunk_message = chunk["choices"][0]["delta"].get("content", "")
            full_text += chunk_message
            message_placeholder.write(full_text)  # プレースホルダーを更新して段階的に表示

# 開催期間を解析し、複数の期間をリストで返すヘルパー関数
def extract_event_dates(event_period_str):
    event_period_str = re.sub(r'（.*?）', '', event_period_str)
    date_ranges = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日)～(\d{1,2}月\d{1,2}日)', event_period_str)
    if not date_ranges:  # 単一日付の場合の処理
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


def background_func(background_info):
    return (f"あなたは以下の背景情報を持つ旅行ガイドです。{background_info}"
                f"あなたは自治体の有能な旅行ガイダンスAIアシスタントで、ミッションは与えられたコンテンツ情報からユーザーにその場所の紹介を行うことです。"
                f"以下の手順と条件をもとに与えられた背景情報"
                f"を要約して観光地を紹介してください。注意、要約してください。すべて背景情報からコピペじゃなくて要約です。分かりやすく読みやすいように！"
                f"重要なキーワードを取りこぼさない。"
                f"架空の表現や言葉を使用しない。"
                f"文章中の数値には変更を加えない。")
                
# Word文書を読み込む関数
def read_word_document(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# テキストからEmbeddingを生成する関数
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return np.array(response['data'][0]['embedding'])

# 文書から最も関連性の高いセクションを見つける関数
def find_most_relevant_section(document_text, query, top_n=1):
    paragraphs = document_text.split('\n')  # 文書を段落ごとに分割
    paragraph_embeddings = [get_embedding(p) for p in paragraphs]  # 段落ごとにEmbeddingを生成
    query_embedding = get_embedding(query)  # クエリのEmbeddingを生成

    # 余弦類似度を計算して関連性を評価
    similarities = [np.dot(query_embedding, p_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(p_emb)) for p_emb
                    in paragraph_embeddings]

    # 最も関連性の高い段落を取得
    most_relevant_idx = np.argsort(similarities)[-top_n:]
    return [paragraphs[i] for i in most_relevant_idx]


def generate_prompt(location, how, time1, time2, user_input):

    user_prompt = f"私に旅行プランを表形式で立ててほしい。行先などの情報は以下の通りです。私は{location}に旅行に行きたい。到着日は{time1}で、帰還日は{time2}です。"

    if how == "レンタカー":
        user_prompt += f"私は地元でレンタカーを借りたいので、レンタカー比較サービスサイトを出してください。"
        user_prompt += f"レンタカー比較サイトとして、https://www.tour.ne.jp/j_rentacar/　を提案してください。"
        user_prompt += f"旅行プランを立ててくれる時は駐車場も注意して提案してください。"
    if how == "マイカー":
        user_prompt += f"旅行プランを立ててくれる時は駐車場も注意して提案してください。"
    if how == "公共交通":
        user_prompt += f"旅行プランを立ててくれる時は公共交通にかかる時間も注意して提案してください。"
    user_prompt += f"時間帯ごとに訪れる場所と各場所の簡単な説明も含めてください。あまりにもバラバラ過ぎた場所にそれぞれ行くんじゃなくてまとめて近い場所を全部訪れてから次に行くように。"
    user_prompt += f"そして、忙し過ぎる日程設定はしないでください。18時以降の参観はなるべく寺社を避けて余裕を持って観光できるようなそんなに忙しくないプランをください。"
    user_prompt += f"以下は要望です："
    return user_prompt + user_input

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()
    return document
        
def main():
    if 'map' not in st.session_state:
        st.session_state.map = None

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "plan" not in st.session_state:  # sidebar+plan?
        st.session_state.plan = ""

    if "satisfaction" not in st.session_state:
        st.session_state.satisfaction = 5 #default

    options = ['スケジュールを立てたい', '時間限定イベントに参加したい']

    # 複数選択可能なプルダウンリストを表示し、選択された項目を取得
    selected_options = st.radio('オプション:', options)

    if selected_options == 'スケジュールを立てたい':
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox('旅行の行先を選択してください:', ("京都", "大阪"))
        with col2:
            how = st.selectbox('旅行の行先を選択してください:', ("マイカー", "レンタカー", "公共交通"))

        col3, col4 = st.columns(2)
        with col3:
            time1 = st.date_input('到着日を選択してください:', datetime.now())
        with col4:
            time2 = st.date_input('帰還日を選択してください:', datetime.now())

        user_input = st.text_area('具体的なニーズをここで述べてください:')

        if user_input and len(user_input) <= 2:
            error_message = "3文字以上入力してください。"
            if error_message:
                st.error(error_message)

        with st.sidebar:
            sidebar_input = st.text_area('質問内容(例：東寺の歴史を教えてください):')

            document_text = read_word_document("./modelcourse.docx")
            background_info = find_most_relevant_section(document_text, sidebar_input)
            system_prompt1 = background_func(background_info)

            is_sidebar_clicked = st.button("質問してみる")
            if is_sidebar_clicked:

                message = [
                {"role": "system", "content": system_prompt1},
                {"role": "user", "content": sidebar_input},
                    ]

                response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                max_tokens=2000,
                temperature=0,
                stream=True
            )
                partial_words = ""

                for chunk in response:
                    if 'choices' in chunk:
                        chunk_message = chunk['choices'][0]['delta'].get('content', '')
                        partial_words += chunk_message

                st.write(partial_words)

        is_generate_clicked = st.button("文章生成")

        if is_generate_clicked:

            if selected_options ==  "スケジュールを立てたい":
                if 'last_response' not in st.session_state:
                    st.session_state.last_response = ""

                question = generate_prompt(location, how, time1, time2, user_input)
                doc = read_word_document("./modelcourse.docx")
                doc_info = find_most_relevant_section(doc, question)
                system_text = (f"あなたは自治体の有能な旅行ガイダンスAIアシスタントで、ミッションはユーザーのニーズに応じて、旅行プランを立てることまたは後述する追加情報を基づいて具体的にユーザーに観光地について教えることです。"
                                f"ユーザーのニーズに応じて答えること。"
                                 f"プランを聞かれた時は余裕のあるプランを立てること、無理やりたくさん行かなくても良く、余裕を持って時間を見積もること。"
                                 f"プランをチャート形式で出すこと。あなたは自分が持っている情報のほか{doc_info}という情報も持っている")

                if len(st.session_state['last_response']) == 0:
                    pass
                else:
                    system_text += "あなたはすでにこのアドバイスをしている、ユーザーからの質問はこれに踏まえて聞いている：" +  st.session_state['last_response']

                message = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": question},
                    ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                max_tokens=2000,
                temperature=0,
                stream=True
            )

            partial_words = ""


        # ストリーミングレスポンスの処理
            for chunk in response:
                if 'choices' in chunk:
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    partial_words += chunk_message

            st.session_state['last_response'] = partial_words
            st.session_state.plan = partial_words  # Store the plan in session_state

            times, locations = extract_time_and_location(partial_words)
            display_map(locations)

            if st.session_state.map:
                folium_static(st.session_state.map, width=500, height=300)
            st.session_state.message_history.append({"role": "assistant", "content": partial_words})

            if st.session_state.plan:
                st.write(st.session_state.plan)
                st.session_state.satisfaction = st.slider('このプランに満足していますか？', 1, 5,
                                                          st.session_state.satisfaction)

        if st.button('Save Screen as PDF'):
            pdf = PDF()
            pdf.add_page()
            pdf.add_font('NotoSans', '', 'NotoSansCJKjp-Regular.ttf', uni=True)
            pdf.set_font('NotoSans', '', 12)
            # Print only the added section
            pdf.multi_cell(0, 10, f'満足度: {st.session_state.satisfaction} 星\n\n')
            pdf.multi_cell(0, 10, st.session_state.plan)

            # Save the generated PDF
            pdf_output_path = 'Your_own_plan.pdf'
            pdf.output(pdf_output_path)

            st.success(f'PDFとして保存しました: {pdf_output_path}')


    elif selected_options == '時間限定イベントに参加したい':
        selected_cities = st.radio('オプション:', ("京都", "大阪"))
        mode = st.radio("表示方法を選択してください", ("イベント名で検索", "日付と予算で検索", "自然言語で検索"))

        if mode == "イベント名で検索":
            # イベント名を選択するためのドロップダウンメニューを表示
            event_names = [event["イベント名"] for event in events_data[selected_cities]]
            selected_event_name = st.selectbox("表示したいイベントを選んでください", event_names)

            if selected_event_name:
                selected_event = next(event for event in events_data[selected_cities] if event["イベント名"] == selected_event_name)
                st.subheader(f"イベント名称: {selected_event['イベント名']}")
                st.write(f"開催地: {selected_event['開催地']}")
                st.write(f"開催期間: {selected_event['開催期間']}")
                st.write(f"料金: {selected_event['料金']}")
                st.write(f"説明: {selected_event['説明']}")

        elif mode == "日付と予算で検索":
            selected_date = st.date_input("来訪予定の日付を選択してください", value=date.today())
            budget = st.number_input("予算を入力してください（円）", min_value=0, value=1000)

            matching_events = []
            for event in events_data[selected_cities]:
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
                    generate_event_recommendation(user_input, events_data[selected_cities])
                else:
                    st.write("入力をしてください。")

if __name__ == "__main__":
    main()
