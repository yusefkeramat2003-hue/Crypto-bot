# bot_runner_pro.py
# pip install pandas numpy requests groq python-dotenv

import os
import time
import json
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ==================== CONFIGURATION ====================
# دسته‌بندی هوشمند ارزها و حساسیت نوسان بر اساس ATR%
COIN_TIERS = {
    # لارج‌کپ / پایدار: نوسان کمتر برای تایید کافی است
    "BTCUSDT": {"tier": "LOW_VOLATILITY", "min_atr_pct": 0.8},
    "ETHUSDT": {"tier": "LOW_VOLATILITY", "min_atr_pct": 1.0},
    "SOLUSDT": {"tier": "MID_VOLATILITY", "min_atr_pct": 1.5},
    "AVAXUSDT": {"tier": "MID_VOLATILITY", "min_atr_pct": 1.8},
    "LINKUSDT": {"tier": "MID_VOLATILITY", "min_atr_pct": 1.8},
    "NEARUSDT": {"tier": "MID_VOLATILITY", "min_atr_pct": 2.0},
    "SUIUSDT":  {"tier": "MID_VOLATILITY", "min_atr_pct": 2.0},
    "APTUSDT":  {"tier": "MID_VOLATILITY", "min_atr_pct": 2.0},
    "ARBUSDT":  {"tier": "MID_VOLATILITY", "min_atr_pct": 2.0},
    "OPUSDT":   {"tier": "MID_VOLATILITY", "min_atr_pct": 2.2},
    # میم‌کوین‌ها / هایپ: نوسان بالاتری برای ورود منطقی نیاز دارند
    "DOGEUSDT": {"tier": "HIGH_VOLATILITY", "min_atr_pct": 2.5},
    "PEPEUSDT": {"tier": "HIGH_VOLATILITY", "min_atr_pct": 3.0},
    "WIFUSDT":  {"tier": "HIGH_VOLATILITY", "min_atr_pct": 3.0},
}

# تایم‌فریم‌های مورد بررسی: ۱ ساعته، ۴ ساعته، ۸ ساعته و روزانه
TIMEFRAMES = ["1h", "4h", "8h", "1d"]
LIMIT = 200

# مدیریت سرمایه (بر مبنای ۱۰۰ دلار)
TOTAL_BALANCE = 100.0
RISK_PCT = 0.01             # ریسک ۱ درصد (۱ دلار در هر معامله)
MAX_POSITION_VALUE = 20.0   # سقف درگیری سرمایه: حداکثر ۲۰ دلار
MIN_RR = 1.3                # حداقل نسبت ریوارد به ریسک

# تنظیمات حلقه و خنک‌سازی
LOOP_INTERVAL_SECONDS = 180 # هر ۳ دقیقه یک چرخه اسکن
COOLDOWN_SECONDS = 3600 * 2  # عدم ارسال سیگنال تکراری برای یک جفت/تایم تا ۲ ساعت

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6981668480").strip()

STATE_FILE = "bot_state.json"
LOG_FILE = "bot_execution.log"


# ==================== HELPERS ====================
def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_signals": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_signals": {}}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"Error saving state: {e}")


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("تلگرام تنظیم نشده؛ پیام فرستاده نشد.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if not res.ok:
            log(f"خطا در ارسال تلگرام: {res.text}")
    except Exception as e:
        log(f"استثنا در ارسال تلگرام: {e}")


# ==================== TECHNICAL INDICATORS ====================
def compute_rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill()


def compute_atr(df: pd.DataFrame, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.bfill()


def enrich(df: pd.DataFrame):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi14"] = compute_rsi(df["close"], 14)
    df["atr14"] = compute_atr(df, 14)
    df["atr_pct"] = (df["atr14"] / df["close"]) * 100
    return df


# ==================== DATA SOURCE (BINANCE) ====================
def fetch_binance_klines(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    tf_map = {
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "8h": "8h",
        "1d": "1d"
    }
    if timeframe not in tf_map:
        raise ValueError(f"تایم‌فریم ناشناخته: {timeframe}")

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": tf_map[timeframe], "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    
    # حذف آخرین کندل (که هنوز بسته نشده و در حال نوسان است) جهت جلوگیری از خطای تحلیلی
    return df.iloc[:-1].copy()


# ==================== PRE-AI SCREENER ====================
def passes_pre_filter(df: pd.DataFrame, min_atr_pct: float) -> bool:
    """
    فیلتر اولیه برای جلوگیری از ارسال داده‌های بی‌کیفیت یا خنثی به هوش مصنوعی
    """
    last = df.iloc[-1]
    # بررسی شرط نوسان حداقلی
    if last["atr_pct"] < min_atr_pct:
        return False
    
    # بررسی وجود مومنتوم یا عدم اشباع شدید در RSI
    rsi = last["rsi14"]
    if rsi < 20 or rsi > 80:  # بازار در حالت افراطی بدون تاییدیه
        return False

    return True


# ==================== AI DECISION (GROQ) ====================
def ask_ai(symbol: str, timeframe: str, df: pd.DataFrame) -> dict:
    if not GROQ_API_KEY:
        return {"signal": "NO_TRADE", "reason": "GROQ_API_KEY ست نشده است"}

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    last_slice = df.tail(60).copy()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "indicators": {
            "close": float(last_slice["close"].iloc[-1]),
            "ema20": float(last_slice["ema20"].iloc[-1]),
            "ema50": float(last_slice["ema50"].iloc[-1]),
            "rsi14": float(last_slice["rsi14"].iloc[-1]),
            "atr_pct": float(last_slice["atr_pct"].iloc[-1]),
        },
        "recent_closes": [float(x) for x in last_slice["close"].tail(10).tolist()]
    }

    system_prompt = (
        "You are an institutional crypto risk-aware trader. "
        "Strict rule: Output ONLY pure valid JSON, no markdown, no explanation outside JSON. "
        "JSON structure: {\"signal\": \"BUY\"|\"SELL\"|\"NO_TRADE\", \"entry\": float, "
        "\"stop_loss\": float, \"take_profit\": float, \"reason\": string}"
    )

    user_prompt = f"Data:\n{json.dumps(payload)}\nIf good R:R trend setup exists, provide levels. Else NO_TRADE."

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("
```"):
text = text.strip("`")
if text.startswith("json"):
text = text[4:].strip()
return json.loads(text)
except Exception as e:
return {"signal": "NO_TRADE", "reason": f"AI Parse Error: {str(e)}"}


# ==================== RISK & POSITION SIZING ====================
def risk_manage(signal_pack: dict, last_price: float):
sig = (signal_pack.get("signal") or "NO_TRADE").upper()
if sig not in ["BUY", "SELL"]:
return {"ok": False, "why": "NO_TRADE"}

try:
entry = float(signal_pack["entry"])
sl = float(signal_pack["stop_loss"])
tp = float(signal_pack["take_profit"])
except Exception:
return {"ok": False, "why": "داده‌های عددی تارگت یا استاپ نامعتبر است"}

if sig == "BUY":
if not (sl < entry < tp):
return {"ok": False, "why": "سطوح ورود/استاپ/تارگت برای BUY همخوانی ندارد"}
risk = entry - sl
reward = tp - entry
else:  # SELL
if not (tp < entry < sl):
return {"ok": False, "why": "سطوح ورود/استاپ/تارگت برای SELL همخوانی ندارد"}
risk = sl - entry
reward = entry - tp

if risk <= 0 or reward <= 0:
return {"ok": False, "why": "میزان ریسک یا سود محاسبه شده صفر است"}

rr = reward / risk
if rr < MIN_RR:
return {"ok": False, "why": f"ریسک به ریوارد ضعیف است: {rr:.2f}"}

# محاسبه حجم بر اساس ریسک ۱ دلار
risk_budget = TOTAL_BALANCE * RISK_PCT
qty = risk_budget / risk
pos_val = qty * entry

# اعمال سقف درگیری ۲۰ دلاری
if pos_val > MAX_POSITION_VALUE:
qty = MAX_POSITION_VALUE / entry
pos_val = MAX_POSITION_VALUE

return {
"ok": True,
"signal": sig,
"entry": entry,
"stop_loss": sl,
"take_profit": tp,
"rr": rr,
"qty": qty,
"position_value": pos_val,
"reason": signal_pack.get("reason", "")
}


# ==================== MAIN SCANNER ENGINE ====================
def run_scan_cycle(state):
log("--- شروع اسکن دوره‌ای بازار ---")
now = int(time.time())

for symbol, meta in COIN_TIERS.items():
for tf in TIMEFRAMES:
key = f"{symbol}_{tf}"
last_sent = state.get("last_signals", {}).get(key, 0)
if (now - last_sent) < COOLDOWN_SECONDS:
continue

try:
df = fetch_binance_klines(symbol, tf, limit=LIMIT)
df = enrich(df)

# فیلتر اولیه قبل از خرج توکن و ارسال به هوش مصنوعی
if not passes_pre_filter(df, meta["min_atr_pct"]):
continue

log(f"نوسان مناسب یافت شد در {symbol} ({tf}) -> ارسال به هوش مصنوعی...")
ai_res = ask_ai(symbol, tf, df)
rm = risk_manage(ai_res, df["close"].iloc[-1])

if not rm["ok"]:
log(f"رد شد در مدیریت ریسک: {symbol} ({tf}) -> {rm['why']}")
continue

# ارسال سیگنال تایید شده به تلگرام
msg = (
f"🎯 **سیگنال معتبر {rm['signal']}**\n\n"
f"🪙 جفت‌ارز: `{symbol}`\n"
f"⏱ تایم‌فریم: `{tf}`\n"
f"🏷 سطح ورود (Entry): `{rm['entry']}`\n"
f"🛑 حد ضرر (Stop Loss): `{rm['stop_loss']}`\n"
f"🎯 حد سود (Take Profit): `{rm['take_profit']}`\n"
f"📊 نسبت سود به ریسک: `{rm['rr']:.2f}`\n"
f"💰 حجم سفارش: `{rm['qty']:.4f}` (~{rm['position_value']:.2f}$)\n\n"
f"💡 دلیل تحلیل: {rm['reason']}"
)
send_telegram(msg)
log(f"سیگنال با موفقیت ارسال شد به تلگرام: {symbol} ({tf})")

# ثبت در حافظه برای جلوگیری از ارسال مجدد در بازه کول‌داون
state.setdefault("last_signals", {})[key] = now
save_state(state)

except Exception as e:
log(f"خطا در پردازش {symbol} ({tf}): {e}")

log("--- پایان دور اسکن ---")


def main():
log("ربات ترید هوشمند Pro راه‌اندازی شد.")
state = load_state()
send_telegram("🚀 ربات اسکنر چندارزی (با تایم‌فریم‌های 1h, 4h, 8h, 1d) با موفقیت روی سرور روشن شد.")

while True:
try:
run_scan_cycle(state)
except Exception as e:
log(f"خطای غیرمنتظره در حلقه اصلی: {e}\n{traceback.format_exc()}")

time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
main()
