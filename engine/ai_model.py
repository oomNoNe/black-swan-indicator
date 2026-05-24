from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        # โหลดโมเดล FinBERT เตรียมไว้
        self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze(self, headline):
        """อ่านพาดหัวข่าวแล้วคืนค่า Sentiment และคะแนนความเสี่ยง"""
        try:
            # ตัดข้อความไม่ให้เกินลิมิต AI
            result = self.nlp(headline[:512])[0]
            label = result['label']

            # แปลงเป็นคะแนน (Negative=100, Neutral=50, Positive=0)
            risk_score = 100 if label == 'negative' else 50 if label == 'neutral' else 0

            return label.upper(), risk_score
        except Exception as e:
            print(f"⚠️ AI Analysis Error: {e}")
            return "NEUTRAL", 50  # กรณี Error ให้ถือว่าเป็นกลางไว้ก่อน