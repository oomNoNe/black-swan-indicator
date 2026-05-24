import pandas as pd
import urllib.request
import xml.etree.ElementTree as ET


def fetch_financial_news():
    """ดึงข่าวการเงินจาก Google News RSS และแปลงเป็น DataFrame เสมอ"""
    try:
        # ใช้ Keyword ค้นหาข่าวการเงินและวิกฤตเศรษฐกิจ
        url = "https://news.google.com/rss/search?q=finance+market+crash+economy&hl=en-US&gl=US&ceid=US:en"

        # ปลอมตัวเป็นเบราว์เซอร์เพื่อไม่ให้ Google บล็อก
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        root = ET.fromstring(response.read())

        news_list = []
        for item in root.findall('.//channel/item')[:10]:  # ดึง 10 ข่าวล่าสุด
            title = item.find('title').text
            pub_date = item.find('pubDate').text

            news_list.append({
                "Source": "Google News",
                "Publish_Date": pub_date,
                "Headline": title
            })

        # 🌟 จุดสำคัญ: แปลง List ให้เป็น Pandas DataFrame ก่อนส่งกลับไปที่ app.py
        df = pd.DataFrame(news_list)
        return df

    except Exception as e:
        print(f"[News Crawler Error] Failed to fetch news: {e}")
        # หากเน็ตหลุดหรือมี Error ให้ส่งตารางเปล่ากลับไป เพื่อป้องกัน app.py พัง
        return pd.DataFrame(columns=["Source", "Publish_Date", "Headline"])