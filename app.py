import math
import datetime as dt
import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)


def cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def rsi14(close):
    d = close.diff().dropna()
    up = d.clip(lower=0).rolling(14).mean().iloc[-1]
    dn = (-d.clip(upper=0)).rolling(14).mean().iloc[-1]
    return 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)


def num(v):
    return v if isinstance(v, (int, float)) and not math.isnan(v) else None


FALLBACK_PEERS = {
    "Utilities": ["NTPC", "POWERGRID", "TATAPOWER", "JSWENERGY", "NHPC", "ADANIGREEN", "TORNTPOWER"],
    "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "COALINDIA", "GAIL"],
    "Financial Services": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE"],
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "Healthcare": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "LUPIN"],
    "Consumer Cyclical": ["MARUTI", "M&M", "TATAMOTORS", "TITAN", "EICHERMOT", "BAJAJ-AUTO"],
    "Consumer Defensive": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "TATACONSUM"],
    "Basic Materials": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "ULTRACEMCO", "GRASIM", "VEDL"],
    "Industrials": ["LT", "SIEMENS", "ABB", "ADANIENT", "BEL", "HAL"],
    "Communication Services": ["BHARTIARTL", "IDEA", "INDUSTOWER"],
    "Real Estate": ["DLF", "GODREJPROP", "LODHA", "OBEROIRLTY", "PRESTIGE"],
}


def live(sym):
    """Latest traded price for one ticker (Yahoo, may lag up to ~15 min)."""
    try:
        fi = yf.Ticker(sym).fast_info
        p, prev = float(fi["last_price"]), float(fi["previous_close"])
        return {"symbol": sym, "exchange": "NSE" if sym.endswith(".NS") else "BSE",
                "price": p, "change": p - prev, "change_pct": (p / prev - 1) * 100}
    except Exception:
        return None


def other_exchange(sym):
    return sym[:-3] + (".BO" if sym.endswith(".NS") else ".NS")


def market_open():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    return now.weekday() < 5 and dt.time(9, 15) <= now.time() <= dt.time(15, 30)


@app.route("/api/quote")
def quote():
    syms = [s for s in request.args.get("symbols", "").split(",") if s][:4]
    return jsonify({"quotes": [q for q in map(live, syms) if q], "market_open": market_open()})


@app.route("/api/peers")
def peers():
    sym = request.args.get("symbol", "").strip()
    base = sym[:-3]
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        info = {}
    cands = []
    try:
        if info.get("industryKey"):
            cands += [s for s in yf.Industry(info["industryKey"]).top_companies.index if str(s).endswith(".NS")]
    except Exception:
        pass
    cands += [s + ".NS" for s in FALLBACK_PEERS.get(info.get("sector"), [])]
    cands = [c for c in dict.fromkeys(cands) if c[:-3] != base][:12]
    if not cands:
        return jsonify([])
    try:
        data = yf.download(cands, period="1y", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception:
        return jsonify([])
    rows = []
    for c in cands:
        try:
            cl = data[c]["Close"].dropna()
            if len(cl) < 130:
                continue
            rr = np.log(cl / cl.shift(1)).dropna()
            r6, r1 = float(cl.iloc[-1] / cl.iloc[-126] - 1), float(cl.iloc[-1] / cl.iloc[0] - 1)
            vol, up = float(rr.std() * math.sqrt(252)), bool(cl.iloc[-1] > cl.tail(200).mean())
            rows.append({"symbol": c, "name": c[:-3], "price": float(cl.iloc[-1]), "ret6m": r6 * 100,
                         "ret1y": r1 * 100, "vol": vol * 100, "trend_up": up,
                         "doing_well": r1 > 0 and up,
                         "score": 0.5 * r6 + 0.3 * r1 - 0.2 * vol + (0.1 if up else 0)})
        except Exception:
            continue
    rows.sort(key=lambda x: -x["score"])
    return jsonify(rows[:4])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        quotes = yf.Search(q, max_results=20).quotes
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 502
    out = []
    for r in quotes:
        s = r.get("symbol", "")
        if s.endswith((".NS", ".BO")):
            out.append({"symbol": s, "exchange": "NSE" if s.endswith(".NS") else "BSE",
                        "name": r.get("shortname") or r.get("longname") or s})
    out.sort(key=lambda x: x["exchange"] != "NSE")
    if not out and q.replace("&", "").replace("-", "").isalnum() and " " not in q:
        out = [{"symbol": q.upper() + ".NS", "exchange": "NSE", "name": q.upper()}]
    return jsonify(out)


@app.route("/api/analyze")
def analyze():
    sym = request.args.get("symbol", "").strip()
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="5y", auto_adjust=True)
        close = hist["Close"].dropna()
        if len(close) < 120:
            return jsonify({"error": "Not enough price history for this symbol."}), 404
        try:
            info = t.info or {}
        except Exception:
            info = {}
    except Exception as e:
        return jsonify({"error": f"Could not fetch data: {e}"}), 502

    lv = live(sym)
    price = lv["price"] if lv else float(close.iloc[-1])
    r = np.log(close / close.shift(1)).dropna()
    vol = float(r.std() * math.sqrt(252))
    mu_hist = float(r.mean() * 252)
    mu = 0.5 * mu_hist + 0.5 * 0.10  # shrink towards a long-run 10% so 5y scenarios stay sane
    mdd = float((close / close.cummax() - 1).min())
    ma50, ma200 = float(close.tail(50).mean()), float(close.tail(200).mean())
    hi52, lo52 = float(close.tail(252).max()), float(close.tail(252).min())
    ret1y = float(close.iloc[-1] / close.iloc[-252] - 1) if len(close) > 252 else None
    rsi = float(rsi14(close))

    beta = None
    try:
        b = yf.Ticker("^NSEI").history(period="5y")["Close"]
        br = np.log(b / b.shift(1)).dropna()
        j = r.to_frame("s").join(br.rename("m"), how="inner").dropna()
        beta = float(j["s"].cov(j["m"]) / j["m"].var())
    except Exception:
        pass

    def horizon(T):
        m, s = (mu - 0.5 * vol ** 2) * T, vol * math.sqrt(T)
        f = lambda z: price * math.exp(m + z * s)
        return {"best": f(1.645), "base": f(0), "worst": f(-1.645),
                "p_loss": cdf(-m / s) * 100}

    h1 = horizon(1)
    var95 = max(0.0, 1 - h1["worst"] / price)
    score = (0.4 * min(vol / 0.6, 1) + 0.3 * min(abs(mdd) / 0.7, 1)
             + 0.2 * min(max(beta if beta is not None else 1, 0) / 1.8, 1)
             + 0.1 * min(var95 / 0.5, 1)) * 100
    label = "Low" if score < 30 else "Moderate" if score < 50 else "High" if score < 70 else "Very high"

    pros, cons = [], []
    pe, roe = num(info.get("trailingPE")), num(info.get("returnOnEquity"))
    de, pm = num(info.get("debtToEquity")), num(info.get("profitMargins"))
    rg = num(info.get("revenueGrowth"))
    if price > ma200: pros.append("Trading above its 200-day average (long-term uptrend).")
    else: cons.append("Trading below its 200-day average (long-term weakness).")
    if ret1y is not None:
        (pros if ret1y > 0.12 else cons if ret1y < 0 else []).append(f"1-year return is {ret1y*100:.1f}%.")
    if roe is not None:
        (pros if roe > 0.15 else cons if roe < 0.08 else []).append(f"Return on equity is {roe*100:.1f}%.")
    if pm is not None:
        (pros if pm > 0.12 else cons if pm < 0.05 else []).append(f"Profit margin is {pm*100:.1f}%.")
    if rg is not None:
        (pros if rg > 0.10 else cons if rg < 0 else []).append(f"Revenue growth is {rg*100:.1f}% year on year.")
    if de is not None:
        (pros if de < 50 else cons if de > 150 else []).append(f"Debt-to-equity is {de/100:.2f}.")
    if pe is not None:
        (pros if 0 < pe < 20 else cons if pe > 40 else []).append(f"P/E ratio is {pe:.1f}.")
    if vol > 0.40: cons.append(f"Very volatile: about {vol*100:.0f}% annualised swings.")
    elif vol < 0.25: pros.append(f"Relatively steady: about {vol*100:.0f}% annualised swings.")
    if mdd < -0.50: cons.append(f"Fell as much as {abs(mdd)*100:.0f}% from a peak in the last 5 years.")
    if rsi > 70: cons.append(f"RSI is {rsi:.0f}: short-term overbought.")
    elif rsi < 30: pros.append(f"RSI is {rsi:.0f}: short-term oversold, a possible bounce.")

    result = {
        "symbol": sym, "name": info.get("longName") or info.get("shortName") or sym,
        "sector": info.get("sector"), "price": price, "currency": info.get("currency", "INR"),
        "hi52": hi52, "lo52": lo52, "ma50": ma50, "ma200": ma200, "rsi": rsi, "pe": pe,
        "vol": vol * 100, "beta": beta, "max_drawdown": mdd * 100, "var95": var95 * 100,
        "risk_score": score, "risk_label": label, "p_loss_1y": h1["p_loss"],
        "scenarios": {"6 months": horizon(0.5), "1 year": h1, "5 years": horizon(5)},
        "pros": pros, "cons": cons,
        "quotes": [q for q in (lv, live(request.args.get("alt") or other_exchange(sym))) if q],
        "market_open": market_open(),
    }

    shares, avg = request.args.get("shares", type=float), request.args.get("avg", type=float)
    pf = request.args.get("portfolio", type=float)
    if shares and shares > 0:
        result["advice"] = advise(result, shares, avg, pf)
    return jsonify(result)


def advise(a, shares, avg, portfolio):
    value = shares * a["price"]
    pl = (a["price"] / avg - 1) * 100 if avg else None
    weight = value / portfolio * 100 if portfolio and portfolio > 0 else None
    down = a["price"] < a["ma200"]
    reasons = []
    if pl is not None: reasons.append(f"You are {'up' if pl >= 0 else 'down'} {abs(pl):.1f}% on this position.")
    if weight is not None: reasons.append(f"It is {weight:.1f}% of your portfolio.")
    reasons.append(f"Risk is {a['risk_label'].lower()} ({a['risk_score']:.0f}/100); the price is {'below' if down else 'above'} its 200-day average.")

    if weight is not None and weight > 25:
        action, why = "Balance it off", "One stock above 25% of your portfolio is concentrated. Trim it and spread the money."
    elif a["risk_score"] >= 50 and down and pl is not None and pl < -15:
        action, why = "Sell (or cut)", "A deep loss, a downtrend and high risk together suggest limiting further damage."
    elif pl is not None and pl > 40 and a["rsi"] > 70:
        action, why = "Sell part (book profit)", "Large gains with an overbought price: lock in some profit and keep the rest."
    elif down and a["rsi"] < 45:
        action, why = "Wait", "The trend is weak. Avoid adding more until the price recovers above its 200-day average."
    else:
        action, why = "Hold", "Nothing here argues for acting now. Review again if the trend or the company's results change."
    return {"action": action, "why": why, "reasons": reasons, "value": value, "pl_pct": pl, "weight": weight}


if __name__ == "__main__":
    app.run(debug=True)
