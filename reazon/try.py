import streamlit as st
import openai
from datetime import datetime
import re



# GPT生成的内容中添加URL链接和图片
def process_gpt_response_with_links_and_images(response_text):
    # 正则表达式查找地名并生成链接和图片
    def generate_link_and_image(location_name):
        # 使用GPT生成地名的URL链接（例如Wikipedia）
        search_url = f"https://en.wikipedia.org/wiki/{location_name.replace(' ', '_')}"
        # 假设我们使用一个通用的图片搜索引擎生成图片链接
        image_url = f"https://source.unsplash.com/featured/?{location_name.replace(' ', '%20')}"
        return f'<a href="{search_url}" target="_blank">{location_name}</a><br><img src="{image_url}" alt="{location_name}" width="300px">'

    # 处理生成的内容，将地名替换为带链接和图片的HTML
    words = response_text.split()
    processed_words = []
    for word in words:
        if re.match(r'[A-Za-z\u4e00-\u9fff]+', word):  # 简单地识别单词或汉字
            processed_words.append(generate_link_and_image(word))
        else:
            processed_words.append(word)
    return ' '.join(processed_words)


def generate_prompt(location, how, time1, time2, user_input):
    user_prompt = (
        f"私に旅行プランを立ててほしい。行先などの情報は以下の通りです。"
        f"私は{location}に旅行に行きたい。到着日は{time1}で、帰還日は{time2}です。"
        "訪問する場所については、各場所の名前をクリック可能なリンクにしてください。"
        "リンク应链接到关于该地点的网页（例如Wikipedia），并包含一个相关的图片。"
    )
    if how == "レンタカー":
        user_prompt += (
            "私は地元でレンタカーを借りたいので、レンタカー比較サービスサイトを出してください。"
            "旅行プランを立ててくれる時は駐車場も注意して提案してください。"
        )
    if how == "マイカー":
        user_prompt += "旅行プランを立ててくれる時は駐車場も注意して提案してください。"
    if how == "公共交通":
        user_prompt += "旅行プランを立ててくれる時は公共交通にかかる時間も注意して提案してください。"
    user_prompt += "時間帯ごとに訪れる場所と各場所の簡単な説明も含めてください。"
    user_prompt += "具体的なニーズを考慮して、旅行プランを立ててください。"
    user_prompt += f"さらに、ユーザーがリクエストしたニーズは次の通りです：{user_input}"
    return user_prompt


def main():
    st.title("旅行プラン生成")

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

    is_generate_clicked = st.button("文章生成")
    if is_generate_clicked:
        question = generate_prompt(location, how, time1, time2, user_input)
        system_text = (
            "あなたは自治体の有能な旅行ガイダンスAIアシスタントで、"
            "ミッションはユーザーのニーズに応じて、旅行プランを立てることです。"
            "ユーザーのニーズに応じて答えること。"
            "余裕のあるプランを立てること、無理やりたくさん行かなくても良く、余裕を持って時間を見積もること。"
            "できるだけ具体的に行先を教えること。"
        )

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

        full_text = ""
        for chunk in response:
            if 'choices' in chunk:
                chunk_message = chunk['choices'][0]['delta'].get('content', '')
                full_text += chunk_message

        # 处理生成的文本，将地名替换为带有链接和图片的版本
        processed_text = process_gpt_response_with_links_and_images(full_text)

        # 显示生成的内容
        st.markdown(processed_text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
