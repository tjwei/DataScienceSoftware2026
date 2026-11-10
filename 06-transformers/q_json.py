def parse_course(text, retries=3):
    prompt = f"""從下面的課程介紹擷取資訊，只輸出 JSON，不要其他文字。格式：
{{"課名": str, "老師": str, "時間": str, "學分": int}}
找不到的欄位填 null。

課程介紹：{text}"""
    for _ in range(retries):          # 解析失敗就重試 (因為有 sampling, 每次結果不同)
        try:
            return extract_json(chat([{"role": "user", "content": prompt}]))
        except Exception as e:
            print("解析失敗，重試:", e)
    return None

for t in ["資料科學軟體與計算實驗，由魏老師授課，每週二 9:00-12:00，3 學分。",
          "本學期開設「深度學習導論」(2學分)，授課教師：王小明，時間週四下午兩點到四點",
          "Machine Learning, Prof. Lee, Fridays 10am, 3 credits"]:
    print(parse_course(t))
