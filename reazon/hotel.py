import openai
import concurrent.futures  # concurrent.futures モジュールをインポート
import numpy as np


# キャッシュ機能
embedding_cache = {}


# テキストファイルを読み込む関数
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# キャッシュ機能付きのテキストからEmbeddingを生成する関数
def get_embedding_with_cache(text, model="text-embedding-ada-002"):
    if text in embedding_cache:
        return embedding_cache[text]
    embedding = get_embedding(text, model)
    embedding_cache[text] = embedding
    return embedding


# テキストからEmbeddingを生成する関数
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return np.array(response['data'][0]['embedding'])


# 文書から最も関連性の高いセクションを見つける関数（並列処理対応）
def find_most_relevant_section(document_text, query, top_n=1):
    paragraphs = document_text.split('\n')
    query_embedding = get_embedding_with_cache(query)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        paragraph_embeddings = list(executor.map(lambda p: get_embedding_with_cache(p), paragraphs))

    similarities = [np.dot(query_embedding, p_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(p_emb)) for p_emb
                    in paragraph_embeddings]
    most_relevant_idx = np.argsort(similarities)[-top_n:]
    return [paragraphs[i] for i in most_relevant_idx]


# ホテル情報を解析する関数
def parse_hotel_information(hotel_text):
    hotels = []
    lines = hotel_text.split('\n')

    # Skip the header
    for line in lines[1:]:
        if line.strip():
            try:
                parts = line.split(',')
                if len(parts) != 8:
                    print(f"解析できない行: {line}")  # 调试信息，帮助定位问题
                    continue

                hotel = {
                    'name': parts[0].strip(),
                    'location': parts[1].strip(),
                    'distance': parts[2].strip(),
                    'score': float(parts[3].strip()),
                    'location_score': float(parts[4].strip()) if parts[4].strip() != 'なし' else None,
                    'room_description': parts[5].strip(),  # 确保房间描述作为字符串处理
                    'original_price': int(parts[6].replace('￥', '').replace(',', '').strip()),  # 解析原始价格
                    'current_price': int(parts[7].replace('￥', '').replace(',', '').strip())  # 解析当前价格
                }
                hotels.append(hotel)
            except (ValueError, IndexError) as e:
                print(f"ホテル情報の解析中にエラーが発生しました: {e}")
                continue
    print(f"解析されたホテル情報: {hotels}")  # 解析されたホテル情報を出力
    return hotels


# ホテルを推薦する関数
def recommend_hotels_by_price(hotels, budget, top_n=5):
    # 予算以内のホテルをフィルタリング
    filtered_hotels = [hotel for hotel in hotels if hotel['current_price'] <= budget]

    # 価格でソート（低い順）
    sorted_hotels = sorted(filtered_hotels, key=lambda x: x['current_price'])

    # 上位N件を返す
    return sorted_hotels[:top_n]


# テキストファイルを読み込み
print("読み込み中...")
hotel_text = read_text_file("C:/Users/Administrator/Desktop/reazon/hotel.txt")
hotels = parse_hotel_information(hotel_text)

# 予算を考慮したホテル推薦
user_budget = 35000  # 例えば、ユーザーが設定した予算
recommended_hotels = recommend_hotels_by_price(hotels, user_budget)

# 結果を出力
print("\n推薦されたホテル（予算を考慮、価格順）:\n")
if recommended_hotels:
    for hotel in recommended_hotels:
        print(f"ホテル名: {hotel['name']}, 位置: {hotel['location']}, 距離: {hotel['distance']}, "
              f"評価: {hotel['score']}/10, 現在の価格: {hotel['current_price']}円, 部屋説明: {hotel['room_description']}")
else:
    print("予算内に該当するホテルがありません。")
