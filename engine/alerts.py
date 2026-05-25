"""
Alert system สำหรับส่ง notification เมื่อ risk score เกิน threshold

รองรับ:
- Discord webhook (แนะนำ — ฟรี, setup ง่าย, ไม่มี deprecation)
- Slack webhook (รูปแบบเดียวกัน)
- Custom webhook URL (POST JSON)

⚠️ Note: Line Notify ถูก deprecated ในปี 2025 — แทนที่ด้วย LINE Messaging API
   ซึ่งต้องสมัคร LINE Developers + Channel access token
   ซับซ้อนกว่าและไม่เหมาะกับ portfolio piece → เราใช้ Discord แทน

วิธีตั้งค่า Discord webhook:
1. Discord Server → Channel → Edit Channel → Integrations → Webhooks
2. New Webhook → Copy URL
3. set env var: DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
"""
import os
import json
import requests
from datetime import datetime


def get_webhook_url():
    """อ่าน webhook URL จาก env (ปลอดภัยกว่าใส่ใน code)"""
    return os.environ.get("DISCORD_WEBHOOK_URL", "").strip()


def format_alert_message(score, vix, regime, predicted_vix=None, threshold=70):
    """สร้าง Discord embed message สวยๆ"""
    # สี: เขียว < 40, ส้ม 40-70, แดง > 70
    color = 0xEF553B if score >= 70 else 0xFFA15A if score >= 40 else 0x00cc96
    severity = "🚨 CRITICAL" if score >= 70 else "⚠️ WARNING" if score >= 40 else "✅ NORMAL"

    embed = {
        "title": f"{severity} — Black Swan Risk Alert",
        "description": f"Crisis Risk Score: **{score:.1f} / 100** (threshold: {threshold})",
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "fields": [
            {"name": "📊 Current VIX", "value": f"{vix:.2f}" if vix else "N/A", "inline": True},
            {"name": "🎭 Market Regime", "value": str(regime), "inline": True},
        ],
        "footer": {"text": "Black Swan Risk Indicator · github.com/oomNoNe/black-swan-indicator"},
    }

    if predicted_vix is not None:
        delta = predicted_vix - vix if vix else 0
        arrow = "📈" if delta > 0 else "📉"
        embed["fields"].append({
            "name": f"🔮 AI Forecast (7d)",
            "value": f"{predicted_vix:.2f} {arrow} ({delta:+.2f})",
            "inline": True,
        })

    return {"embeds": [embed]}


def send_alert(score, vix, regime, predicted_vix=None, threshold=70,
               webhook_url=None, force=False):
    """
    ส่ง alert ผ่าน Discord webhook

    Args:
        score: crisis risk score (0-100)
        vix: current VIX value
        regime: market regime string
        predicted_vix: optional 7-day forecast
        threshold: ส่ง alert เมื่อ score >= threshold (default 70)
        webhook_url: override env var (สำหรับ testing)
        force: ส่งแม้ score ต่ำกว่า threshold (สำหรับ test button)

    Returns:
        dict: {"status": "sent" | "skipped" | "error", "message": str}
    """
    if score < threshold and not force:
        return {"status": "skipped", "message": f"Score {score:.1f} < threshold {threshold}"}

    url = webhook_url or get_webhook_url()
    if not url:
        return {"status": "error", "message": "DISCORD_WEBHOOK_URL not configured"}

    if not url.startswith("https://"):
        return {"status": "error", "message": "Invalid webhook URL (must start with https://)"}

    payload = format_alert_message(score, vix, regime, predicted_vix, threshold)

    try:
        response = requests.post(
            url, data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code in (200, 204):
            return {"status": "sent", "message": f"Alert sent (HTTP {response.status_code})"}
        return {"status": "error",
                "message": f"Webhook returned HTTP {response.status_code}: {response.text[:200]}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Webhook timeout (10s)"}
    except Exception as e:
        return {"status": "error", "message": f"Webhook error: {e}"}


def test_webhook(webhook_url=None):
    """ส่ง test message — ใช้ใน UI test button"""
    return send_alert(
        score=99.9, vix=42.0, regime="Panic",
        predicted_vix=48.5, threshold=0,
        webhook_url=webhook_url, force=True
    )
