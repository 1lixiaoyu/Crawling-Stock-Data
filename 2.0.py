import tushare as ts
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os

# ====== 配置 ======
TOKEN = "你的token"  # 填你的token
ts.set_token(TOKEN)
pro = ts.pro_api()

os.makedirs("factor_data", exist_ok=True)  # 保存路径

# ====== 获取全部A股列表 ======
print("正在获取全部A股列表...")
stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
codes = stock_list["ts_code"].tolist()
print(f"共找到 {len(codes)} 支A股股票")

# ====== 获取单个股票因子数据 ======
def fetch_factors(ts_code):
    try:
        # 市值、估值等基础数据
        basic = pro.daily_basic(ts_code=ts_code, fields="trade_date,ts_code,total_mv,pe,pe_ttm,pb,turnover_rate")
        if basic.empty:
            return None

        # 取最新一行
        latest_basic = basic.sort_values("trade_date", ascending=False).iloc[0]

        # 对数市值
        ln_mv = np.log(latest_basic["total_mv"] * 1e6)  # total_mv单位亿

        # 盈利质量（ROE、毛利率）
        fina = pro.fina_indicator(ts_code=ts_code, fields="ts_code,roe,profit_gr,op_of_gr,grossprofit_margin")
        fina_latest = fina.iloc[0] if not fina.empty else pd.Series({"roe": np.nan, "grossprofit_margin": np.nan})

        # 成长性（营收增长率、盈利增长率）
        growth = pro.fina_indicator(ts_code=ts_code, fields="ts_code,or_yoy,np_yoy")
        growth_latest = growth.iloc[0] if not growth.empty else pd.Series({"or_yoy": np.nan, "np_yoy": np.nan})

        # 动量（过去12个月收益率剔除最近1个月）
        daily_price = pro.daily(ts_code=ts_code, fields="trade_date,close")
        daily_price["trade_date"] = pd.to_datetime(daily_price["trade_date"])
        daily_price = daily_price.sort_values("trade_date")
        if len(daily_price) >= 252:
            price_12m = daily_price.iloc[-252]["close"]
            price_1m = daily_price.iloc[-21]["close"]
            price_now = daily_price.iloc[-1]["close"]
            momentum = (price_now / price_12m - 1) - (price_now / price_1m - 1)
        else:
            momentum = np.nan

        # 波动率（过去一年日收益率标准差）
        daily_price["ret"] = daily_price["close"].pct_change()
        volatility = daily_price["ret"].std() * np.sqrt(252) if len(daily_price) > 2 else np.nan

        # 流动性（Amihud 非流动性指标）
        money_flow = pro.moneyflow(ts_code=ts_code, fields="buy_sm_vol,sell_sm_vol,buy_md_vol,sell_md_vol,buy_lg_vol,sell_lg_vol,buy_elg_vol,sell_elg_vol")
        amihud = np.nan
        if not daily_price.empty:
            amihud = (np.abs(daily_price["ret"]) / (latest_basic["total_mv"] * 1e6)).mean()

        return {
            "ts_code": ts_code,
            "ln_mv": ln_mv,
            "pe": latest_basic["pe"],
            "pe_ttm": latest_basic["pe_ttm"],
            "pb": latest_basic["pb"],
            "momentum_12m_ex1m": momentum,
            "volatility": volatility,
            "turnover_rate": latest_basic["turnover_rate"],
            "amihud": amihud,
            "roe": fina_latest.get("roe"),
            "grossprofit_margin": fina_latest.get("grossprofit_margin"),
            "revenue_growth": growth_latest.get("or_yoy"),
            "profit_growth": growth_latest.get("np_yoy")
        }

    except Exception as e:
        return None

# ====== 多线程爬取 ======
factors_data = []
with ThreadPoolExecutor(max_workers=8) as executor:  # 控制线程数避免超限
    futures = {executor.submit(fetch_factors, code): code for code in codes}
    for future in tqdm(as_completed(futures), total=len(futures), desc="获取因子数据"):
        result = future.result()
        if result:
            factors_data.append(result)

# ====== 保存结果 ======
df_factors = pd.DataFrame(factors_data)
df_factors.to_csv("factor_data/all_factors.csv", index=False, encoding="utf-8-sig")

# 按因子拆分保存
for col in df_factors.columns:
    if col != "ts_code":
        df_factors[["ts_code", col]].to_csv(f"factor_data/{col}.csv", index=False, encoding="utf-8-sig")

print("\n✅ 全部因子数据已获取并保存到 factor_data 目录")
