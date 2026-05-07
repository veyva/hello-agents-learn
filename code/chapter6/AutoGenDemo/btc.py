# btc_price_app.py

import requests
import streamlit as st
import logging
from datetime import datetime

# ==================
# 日志设置
# ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ==================
# API常量
# ==================
COINGECKO_SIMPLE_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
)
COINGECKO_MARKET_CHART = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    "?vs_currency=usd&days=1"
)

def get_bitcoin_data(timeout=10):
    """获取比特币当前价格、24h涨跌百分比和涨跌额"""
    try:
        # 获取实时价格与24h涨跌幅
        resp = requests.get(COINGECKO_SIMPLE_URL, timeout=timeout)
        resp.raise_for_status()
        r = resp.json().get("bitcoin", {})
        price = float(r.get("usd", 0.0))
        change_percent = float(r.get("usd_24h_change", 0.0))

        # 获取24小时前的价格
        history_resp = requests.get(COINGECKO_MARKET_CHART, timeout=timeout)
        history_resp.raise_for_status()
        prices = history_resp.json().get("prices", [])
        price_24h_ago = prices[0][1] if prices and len(prices[0]) >= 2 else None
        if price_24h_ago is None:
            raise ValueError("无法获取24小时前价格")
        change_abs = price - price_24h_ago

        return dict(
            price=price,
            change_percent=change_percent,
            change_abs=change_abs,
            price_24h_ago=price_24h_ago,
        )
    except Exception as e:
        logging.exception("数据获取失败")
        raise RuntimeError(f"数据获取失败: {e}")

def show_price_data(data: dict):
    """显示价格、涨跌额/幅度（颜色美化、格式化输出）"""
    price = data["price"]
    change_percent = data["change_percent"]
    change_abs = data["change_abs"]
    price_24h_ago = data["price_24h_ago"]
    arrow = "⬆️" if change_abs >= 0 else "⬇️"
    change_color = "green" if change_abs >= 0 else "red"

    st.markdown(
        f"""
        <div style="text-align:center;">
            <h1>₿ 比特币 (BTC)</h1>
            <h2 style="font-size:2.8rem; margin-top:0.5em;">
                ${price:,.2f}
            </h2>
            <div style="font-size:1.2rem; margin-bottom:0.7em; color:gray;">
                24H前: ${price_24h_ago:,.2f}
            </div>
            <span style="color:{change_color}; font-size:1.5rem; font-weight:bold;">
                {arrow} {change_abs:+,.2f} ({change_percent:+.2f}%)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        page_title="比特币价格",
        page_icon="₿",
        layout="centered",
    )

    st.title('📈 比特币价格展示 🪙')
    st.caption('数据来源 [CoinGecko](https://www.coingecko.com/)')

    refresh = st.button("刷新", type="primary")

    if "btc_data" not in st.session_state:
        st.session_state["btc_data"] = None
        st.session_state["last_update"] = None
        st.session_state["last_error"] = None

    # 需要重新拉取
    if refresh or st.session_state["btc_data"] is None:
        with st.spinner("行情数据获取中..."):
            try:
                btc_data = get_bitcoin_data()
                st.session_state["btc_data"] = btc_data
                st.session_state["last_update"] = datetime.now()
                st.session_state["last_error"] = None
            except Exception as e:
                st.session_state["btc_data"] = None
                st.session_state["last_error"] = str(e)

    # 展示
    if st.session_state.get("btc_data"):
        show_price_data(st.session_state["btc_data"])
        update_time = st.session_state.get("last_update")
        if update_time:
            st.info(f"最后更新时间：{update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        error = st.session_state.get("last_error")
        st.error(error or "暂时无法获取数据，请稍后再试")

    # UI 美化部分（按钮宽度、字号）
    st.markdown(
        """
        <style>
            .stButton > button {
                width: 100%;
                font-size: 1.15rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("---")
    st.markdown("💡 本应用仅供学习与行情参考，不构成投资建议。")

if __name__ == "__main__":
    main()