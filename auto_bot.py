"""
A股选股与异动监控 自动化脚本 (V5.2 动态风控与盈亏比过滤)
专为 GitHub Actions 云端自动化推送设计

技术栈:
- 数据获取: AkShare / 直接HTTP请求
- 数据处理: Pandas
- AI 智能研判: OpenAI SDK (兼容国内大模型)
- 推送通知: Bark
- 技术指标: MACD, KDJ, BOLL
- 风控模块: 动态星级风控 + 仓位硬限拦截 + 盈亏比过滤
"""

import os
import time
import math
import akshare as ak
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor

# ==================== 禁用代理设置 ====================
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# ==================== 环境变量配置 ====================
API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE", "https://llmapi.paratera.com/v1/")
MODEL_NAME = os.environ.get("MODEL_NAME", "DeepSeek-R1")
BARK_URL = os.environ.get("BARK_URL", "")

# ==================== 配置参数 ====================
# 第一阶段筛选配置 (14:30 基础量价与资金初筛)
PHASE_1_CONFIG = {
    'min_change': 3.0,              # 最小涨幅 % (3.0%)
    'max_change': 5.0,               # 最大涨幅 % (5.0%)
    'min_market_cap': 5000000000,   # 最小流通市值 50亿
    'max_market_cap': 20000000000,  # 最大流通市值 200亿
    'min_volume_ratio': 1.2,        # 最小量比
    'max_volume_ratio': 5.0,        # 最大量比
    'min_turnover': 5.0,             # 最小换手率 %
    'max_turnover': 10.0,            # 最大换手率 %
}

# 第二阶段筛选配置 (技术形态与突破确认)
PHASE_2_CONFIG = {
    'history_days': 40,              # 历史数据天数
    'ma_periods': [5, 10, 20],      # 均线周期
    'breakout_threshold': 0.95,     # 突破阈值（20日最高价的95%）
    'chase_threshold': 0.08,        # 拒绝追高阈值（距MA5涨幅 <= 8%）
}

# 历史数据缓存
stock_history_cache = {}


# ==================== Bark 推送函数 ====================
def send_bark(title, body):
    if not BARK_URL:
        print("未配置 BARK_URL，跳过推送")
        return
    try:
        # 去除末尾的斜杠，得到基础 URL
        base_url = BARK_URL.rstrip('/')
        
        # 将文字打包成 JSON 数据发送，彻底突破 GET 请求的 414 长度限制
        payload = {
            "title": title,
            "body": body,
            "isArchive": 1,
            "icon": "https://cdn-icons-png.flaticon.com/512/2942/2942263.png"
        }
        
        # 使用 POST 请求发送数据包
        res = requests.post(base_url, json=payload, timeout=10)
        print(f"Bark 推送状态: {res.status_code}")
        
        if res.status_code != 200:
            print(f"Bark 返回报错详情: {res.text}")
            
    except Exception as e:
        print(f"Bark 推送发生异常: {e}")


# ==================== 数据获取函数 ====================

def get_realtime_data_tencent():
    """
    使用腾讯接口获取A股实时行情数据
    腾讯接口提供完整的数据包括换手率、量比等
    """
    try:
        url = "http://qt.gtimg.cn/q"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': 'http://gu.qq.com/',
            'Connection': 'keep-alive',
        }

        code_ranges = [
            (600000, 606000, 'sh'),  # 沪市主板
            (688000, 690000, 'sh'),  # 科创板
            (0, 4000, 'sz'),         # 深市主板+中小板
            (300000, 303000, 'sz'),  # 创业板
        ]

        stocks_data = []
        batch_size = 200
        total_batches = 0

        for start, end, market in code_ranges:
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                batch_codes = [f"{market}{i:06d}" for i in range(batch_start, batch_end)]
                all_codes = ','.join(batch_codes)

                full_url = f"{url}={all_codes}"

                try:
                    response = requests.get(full_url, headers=headers, timeout=10)
                    if response.status_code != 200:
                        continue

                    content = response.text
                    pattern = r'v_([a-z]+)(\d+)="([^"]+)"'
                    matches = re.findall(pattern, content)

                    for match in matches:
                        mk, code, data = match
                        if not data or data.strip() == '':
                            continue

                        parts = data.split('~')

                        if len(parts) >= 50 and parts[1]:
                            try:
                                name = parts[1]
                                price = float(parts[3]) if parts[3] else 0
                                prev_close = float(parts[4]) if parts[4] else 0
                                change_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0

                                volume = 0
                                try:
                                    if parts[36]:
                                        volume = float(parts[36]) * 100
                                except (ValueError, IndexError):
                                    pass

                                amount = 0
                                try:
                                    if parts[37]:
                                        amount = float(parts[37]) * 10000
                                except (ValueError, IndexError):
                                    pass

                                market_cap = 0
                                try:
                                    if parts[44]:
                                        market_cap = float(parts[44]) * 100000000
                                except (ValueError, IndexError):
                                    pass

                                turnover = 0
                                try:
                                    if parts[38]:
                                        turnover = float(parts[38])
                                except (ValueError, IndexError):
                                    pass

                                volume_ratio = 0
                                try:
                                    if parts[49]:
                                        volume_ratio = float(parts[49])
                                except (ValueError, IndexError):
                                    pass

                                vwap = price
                                if volume > 0 and amount > 0:
                                    try:
                                        vwap = amount / volume
                                    except ZeroDivisionError:
                                        pass

                                pe = 0
                                try:
                                    if len(parts) > 39 and parts[39]:
                                        pe = float(parts[39])
                                except (ValueError, IndexError):
                                    pass

                                stocks_data.append({
                                    '代码': f"{mk}{code.zfill(6)}",
                                    '名称': name,
                                    '最新价': price,
                                    '涨跌幅': change_pct,
                                    '成交量': volume,
                                    '成交额': amount,
                                    '换手率': turnover,
                                    '量比': volume_ratio,
                                    '流通市值': market_cap,
                                    '日内均价': vwap,
                                    '市盈率': pe,
                                })
                            except (ValueError, ZeroDivisionError, IndexError):
                                continue

                    total_batches += 1
                    if total_batches % 10 == 0:
                        print(f"腾讯接口已处理 {total_batches} 批次，当前获取 {len(stocks_data)} 只股票")

                except Exception as e:
                    continue

        if stocks_data:
            df = pd.DataFrame(stocks_data)
            df = df.drop_duplicates(subset=['代码'], keep='first')
            print(f"腾讯接口获取数据成功，共 {len(df)} 只股票 (分{total_batches}批次)")
            return df
        else:
            print(f"腾讯接口未返回有效数据")
            return None

    except Exception as e:
        print(f"腾讯接口获取数据失败: {e}")
        return None


def get_realtime_data_sina_new():
    """使用新浪最新版接口获取A股实时行情数据"""
    try:
        url = "http://hq.sinajs.cn/list"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': 'http://finance.sina.com.cn/',
            'Connection': 'keep-alive',
        }

        code_ranges = [
            (600000, 606000, 'sh'),
            (688000, 690000, 'sh'),
            (0, 4000, 'sz'),
            (300000, 303000, 'sz'),
        ]

        stocks_data = []
        batch_size = 200
        total_batches = 0

        for start, end, market in code_ranges:
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                batch_codes = [f"{market}{i:06d}" for i in range(batch_start, batch_end)]
                all_codes = ','.join(batch_codes)

                full_url = f"{url}={all_codes}"

                try:
                    response = requests.get(full_url, headers=headers, timeout=10)
                    if response.status_code != 200:
                        continue

                    content = response.text
                    pattern = r'hq_str_([a-z]+)(\d+)="([^"]*)"'
                    matches = re.findall(pattern, content)

                    for match in matches:
                        mk, code, data = match
                        parts = data.split(',')
                        if len(parts) >= 32 and parts[0]:
                            try:
                                price = float(parts[3]) if parts[3] else 0
                                prev_close = float(parts[2]) if parts[2] else 0
                                change_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0

                                stocks_data.append({
                                    '代码': f"{mk}{code.zfill(6)}",
                                    '名称': parts[0],
                                    '最新价': price,
                                    '涨跌幅': change_pct,
                                    '成交量': float(parts[8]) if parts[8] else 0,
                                    '成交额': float(parts[9]) if parts[9] else 0,
                                    '换手率': 0,
                                    '量比': 0,
                                    '流通市值': float(parts[9]) * 10000 if parts[9] else 0,
                                })
                            except (ValueError, ZeroDivisionError):
                                continue

                    total_batches += 1
                    if total_batches % 10 == 0:
                        print(f"已处理 {total_batches} 批次，当前获取 {len(stocks_data)} 只股票")

                except Exception as e:
                    continue

        if stocks_data:
            df = pd.DataFrame(stocks_data)
            print(f"新浪接口获取数据成功，共 {len(df)} 只股票 (分{total_batches}批次)")
            return df
        else:
            print(f"新浪接口未返回有效数据")
            return None

    except Exception as e:
        print(f"新浪接口获取数据失败: {e}")
        return None


def get_realtime_data_from_akshare(max_retries=3):
    """尝试使用 AkShare 获取数据"""
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_spot_em()

            if df.empty:
                print(f"AkShare 返回空数据 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

            numeric_fields = ['最新价', '涨跌幅', '成交量', '成交额', '换手率', '量比', '流通市值']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')

            if '成交量' in df.columns and '成交额' in df.columns:
                df['日内均价'] = (df['成交额'] * 100) / df['成交量']
                df['日内均价'] = df['日内均价'].replace([np.inf, -np.inf], np.nan)

            df = df.dropna(subset=['最新价', '涨跌幅', '成交量'])

            print(f"AkShare 获取数据成功，共 {len(df)} 只股票")
            return df

        except Exception as e:
            error_msg = str(e)
            print(f"AkShare 获取数据失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

            is_network_error = any(x in error_msg.lower() for x in [
                'connection', 'timeout', 'remote', 'abort', 'disconnect'
            ])

            if is_network_error and attempt < max_retries - 1:
                print(f"网络错误，正在重试...")
                time.sleep(1)
                continue

            return None
    return None


def get_realtime_data():
    """
    获取A股实时行情数据 - 多接口备用方案
    尝试顺序: 腾讯接口 -> AkShare -> 新浪接口
    """
    print("尝试腾讯接口（完整数据源）...")
    time.sleep(0.5)

    df = get_realtime_data_tencent()
    if df is not None and not df.empty:
        print("腾讯接口数据获取成功")
        return df

    print("尝试 AkShare 东方财富接口...")
    time.sleep(0.5)

    df = get_realtime_data_from_akshare()
    if df is not None and not df.empty:
        print("AkShare 数据获取成功")
        return df

    print("尝试新浪接口...")
    time.sleep(0.5)

    df = get_realtime_data_sina_new()
    if df is not None and not df.empty:
        print("新浪接口成功，但数据不完整（缺少换手率、量比等）")
        return df

    print("所有数据源均连接失败")
    return None


def get_stock_history(stock_code, days=40, max_retries=2):
    """
    终极直连版：绕过 AkShare，直接调用腾讯官方历史 K 线 JSON 接口
    自带前复权 (qfq)，对海外 IP 极其友好
    """
    if stock_code in stock_history_cache:
        return stock_history_cache[stock_code]

    for attempt in range(max_retries):
        try:
            match = re.search(r'\d{6}', str(stock_code))
            if not match:
                return None
            pure_code = match.group()

            if pure_code.startswith('6'):
                symbol = f"sh{pure_code}"
            elif pure_code.startswith('8') or pure_code.startswith('4'):
                symbol = f"bj{pure_code}"
            else:
                symbol = f"sz{pure_code}"

            url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{days},qfq"
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                continue

            data = response.json()
            if data.get("code") != 0 or symbol not in data.get("data", {}):
                continue

            stock_data = data["data"][symbol]
            kline_list = stock_data.get("qfqday", stock_data.get("day", []))

            if not kline_list:
                continue

            safe_kline_list = [item[:6] for item in kline_list]
            df = pd.DataFrame(safe_kline_list, columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"])

            numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['收盘'])
            df['日期'] = pd.to_datetime(df['日期'])

            for period in PHASE_2_CONFIG['ma_periods']:
                df[f'MA{period}'] = df['收盘'].rolling(window=period).mean()

            df['20日最高'] = df['最高'].rolling(window=20, min_periods=20).max()

            # MACD (12, 26, 9)
            df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
            df['DIF'] = df['EMA12'] - df['EMA26']
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD'] = (df['DIF'] - df['DEA']) * 2

            # KDJ (9, 3, 3)
            low_list = df['最低'].rolling(9, min_periods=1).min()
            high_list = df['最高'].rolling(9, min_periods=1).max()
            rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
            df['K'] = rsv.ewm(com=2, adjust=False).mean()
            df['D_KDJ'] = df['K'].ewm(com=2, adjust=False).mean()
            df['J'] = 3 * df['K'] - 2 * df['D_KDJ']

            # BOLL (20, 2)
            df['BOLL_MID'] = df['MA20']
            df['BOLL_STD'] = df['收盘'].rolling(20).std(ddof=0)
            df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
            df['BOLL_DOWN'] = df['BOLL_MID'] - 2 * df['BOLL_STD']
            df['BOLL_WIDTH'] = df['BOLL_UP'] - df['BOLL_DOWN']

            # OBV
            df['OBV'] = np.where(df['收盘'] > df['收盘'].shift(1), df['成交量'], 
            np.where(df['收盘'] < df['收盘'].shift(1), -df['成交量'], 0)).cumsum()

            # OBV_MA20
            df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

            # TR & ATR
            df['TR'] = np.maximum(df['最高'] - df['最低'], 
                          np.maximum(abs(df['最高'] - df['收盘'].shift(1)), 
                                     abs(df['最低'] - df['收盘'].shift(1))))
            df['ATR_14'] = df['TR'].rolling(window=14).mean()
            
            # 计算 5 日平均振幅，用于判断突破前的波动率是否收敛
            df['振幅'] = (df['最高'] - df['最低']) / df['收盘'].shift(1)
            df['历史振幅_MA5'] = df['振幅'].shift(1).rolling(5).mean()

            stock_history_cache[stock_code] = df
            return df

        except Exception as e:
            time.sleep(0.5)

    return None


def fetch_sectors_for_filtered(df):
    """只为通过初筛的几只股票查询板块"""
    if df.empty:
        df['所属板块'] = '未知'
        return df

    def get_sector(code):
        code_6 = str(code)[-6:]

        # 方案 1: 巨潮资讯 API
        try:
            profile = ak.stock_profile_cninfo(symbol=code_6)
            if not profile.empty and '所属行业' in profile.columns:
                sector = str(profile['所属行业'].iloc[0])
                if '-' in sector:
                    sector = sector.split('-')[0]
                if sector and sector != 'nan':
                    return sector.strip()
        except:
            pass

        # 方案 2: 网易 F10 网页暴力解析
        try:
            url = f"http://quotes.money.163.com/f10/gszl_{code_6}.html"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=3)
            match = re.search(r'所属行业.*?<td[^>]*>(.*?)</td>', res.text, re.S)
            if match:
                sector = match.group(1).strip()
                sector = re.sub(r'<[^>]+>', '', sector).strip()
                if sector and sector != '--':
                    return sector
        except:
            pass

        return '未知'

    with ThreadPoolExecutor(max_workers=5) as executor:
        sectors = list(executor.map(get_sector, df['代码'].tolist()))

    df['所属板块'] = sectors
    return df


# ==================== 第一阶段筛选：一筛 ====================
def screen_phase_1(df):
    """
    第一阶段筛选：量价与资金初筛
    筛选条件（必须全部满足）:
    - 3.0 <= 涨跌幅 <= 5.0
    - 50000000000 <= 流通市值 <= 200000000000 (50亿~200亿)
    - 最新价 > 日内均价 (日内强势)
    - 量比 > 1.2
    - 5.0 <= 换手率 <= 10.0
    - 剔除名称中包含 "ST" 或 "退" 的股票
    """
    config = PHASE_1_CONFIG

    if '名称' in df.columns:
        df_clean = df[~df['名称'].str.contains('ST|退|暂停', na=False, regex=True)].copy()
    else:
        df_clean = df.copy()

    print(f"\n=== 第一阶段筛选调试 ===")
    print(f"初始股票数: {len(df)}")
    print(f"剔除ST后: {len(df_clean)}")

    conditions = (
        (df_clean['涨跌幅'] >= config['min_change']) &
        (df_clean['涨跌幅'] <= config['max_change'])
    )
    print(f"条件1 - 涨跌幅 {config['min_change']}%~{config['max_change']}%: {conditions.sum()} 只")

    if '流通市值' in df_clean.columns:
        cap_condition = (
            (df_clean['流通市值'] >= config['min_market_cap']) &
            (df_clean['流通市值'] <= config['max_market_cap'])
        )
        conditions = conditions & cap_condition
        print(f"条件2 - 流通市值 {config['min_market_cap']/100000000:.0f}~{config['max_market_cap']/100000000:.0f}亿: {cap_condition.sum()} 只")
    print(f"条件1+2后: {conditions.sum()} 只")

    if '量比' in df_clean.columns and (df_clean['量比'] > 0).any():
        vr_condition = (
            (df_clean['量比'] > config['min_volume_ratio']) & 
            (df_clean['量比'] <= config.get('max_volume_ratio', 5.0))
        )
        conditions = conditions & vr_condition
        print(f"条件3 - 量比 {config['min_volume_ratio']} ~ {config['max_volume_ratio']}: {vr_condition.sum()} 只")
    print(f"条件1+2+3后: {conditions.sum()} 只")

    if '换手率' in df_clean.columns and (df_clean['换手率'] > 0).any():
        tr_condition = (
            (df_clean['换手率'] >= config['min_turnover']) &
            (df_clean['换手率'] <= config['max_turnover'])
        )
        conditions = conditions & tr_condition
        print(f"条件4 - 换手率 {config['min_turnover']}%~{config['max_turnover']}%: {tr_condition.sum()} 只")
    print(f"条件1+2+3+4后: {conditions.sum()} 只")

    if '日内均价' in df_clean.columns:
        vwap_condition = (df_clean['最新价'] >= df_clean['日内均价'] * 0.995)
        conditions = conditions & vwap_condition
        print(f"条件5 - 最新价站上日内均价: {vwap_condition.sum()} 只")
    print(f"条件1+2+3+4+5后: {conditions.sum()} 只")

    if '市盈率' in df_clean.columns:
        pe_condition = (df_clean['市盈率'] > 0) & (df_clean['市盈率'] <= 150)
        conditions = conditions & pe_condition
        print(f"条件6 - 市盈率排雷 (0 < PE <= 150): {pe_condition.sum()} 只")
    print(f"最终通过: {conditions.sum()} 只")

    filtered = df_clean[conditions].copy()
    filtered = fetch_sectors_for_filtered(filtered)

    return filtered.sort_values('涨跌幅', ascending=False)


def calculate_sector_resonance(phase_1_df):
    """计算板块共振雷达"""
    if phase_1_df.empty or '所属板块' not in phase_1_df.columns:
        return {}

    sector_count = phase_1_df['所属板块'].value_counts().to_dict()
    active_sectors = {
        sector: count for sector, count in sorted(
            sector_count.items(),
            key=lambda x: x[1],
            reverse=True
        ) if count >= 2 and sector != '未知'
    }

    return active_sectors


# ==================== 第二阶段筛选：二筛 ====================
def screen_phase_2(phase_1_df):
    """
    第二阶段筛选：技术形态与突破确认 (V5.2 弹性打分制)
    """
    config = PHASE_2_CONFIG
    qualified_stocks = []
    failed_stocks = []

    print(f"\n=== 第二阶段筛选开始 ===")
    print(f"待筛选股票数: {len(phase_1_df)}")

    for idx, stock in enumerate(phase_1_df.itertuples()):
        stock_code = stock.代码
        stock_name = stock.名称 if hasattr(stock, '名称') else stock.代码

        print(f"正在扫描: {stock_name}({stock_code}) ({idx + 1}/{len(phase_1_df)})")

        history_df = get_stock_history(stock_code, config['history_days'])

        if history_df is None or history_df.empty or len(history_df) < 20:
            failed_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '失败原因': '历史数据不足或获取失败'
            })
            continue

        latest = history_df.iloc[-1]

        required_fields = ['MA5', 'MA10', 'MA20', '20日最高', 'DIF', 'DEA', 'K', 'D_KDJ', 'BOLL_UP', 'BOLL_WIDTH']
        if any(pd.isna(latest.get(field, None)) for field in required_fields):
            failed_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '失败原因': '技术指标计算不完整'
            })
            continue

        if len(history_df) < 2:
            failed_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '失败原因': '历史数据不足'
            })
            continue
        yesterday = history_df.iloc[-2]

        if pd.isna(yesterday.get('BOLL_WIDTH', None)):
            failed_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '失败原因': 'BOLL指标计算不完整'
            })
            continue

        # 条件A【均线多头】: 严格对齐同花顺：只需收盘价站上三根均线
        condition_a = (latest['收盘'] > latest['MA5']) and (latest['收盘'] > latest['MA10']) and (latest['收盘'] > latest['MA20'])

        # 条件B【量价齐升】: 今日成交量 > 昨日成交量 且 最新价 > 昨日收盘价
        condition_b = (latest['成交量'] > yesterday['成交量']) and (latest['收盘'] > yesterday['收盘'])

        # 条件C【有效突破】: 最新价 >= 过去20个交易日最高价的95%
        condition_c = latest['收盘'] >= (latest['20日最高'] * config['breakout_threshold'])

        # 条件D【拒绝追高】: (最新价 - MA5) / MA5 <= 0.08
        condition_d = (latest['收盘'] - latest['MA5']) / latest['MA5'] <= config['chase_threshold']

        # 条件E【MACD金叉】: DIF > DEA
        condition_e = latest['DIF'] > latest['DEA']

        # 条件F【KDJ金叉】: K > D (KDJ)
        condition_f = latest['K'] > latest['D_KDJ']

        # 条件G【BOLL触轨放大】: 收盘价 >= BOLL上轨95% 且 喇叭口今日 > 昨日
        condition_g = (latest['收盘'] >= latest['BOLL_UP'] * 0.95) and (latest['BOLL_WIDTH'] > yesterday['BOLL_WIDTH'])

        # 条件 H【OBV 资金潜伏】: OBV 必须站上 20 日均线，说明主力近期是净买入状态
        condition_h = latest['OBV'] > latest['OBV_MA20']
        
        # 条件 I【VCP 蓄力突破】: 突破前5天的平均振幅必须小于 5%（弹簧压紧），且今日量比大于昨日（爆发）
        condition_i = (latest['历史振幅_MA5'] < 0.05) and (latest['成交量'] > yesterday['成交量'] * 1.5)
        
        # 条件 J【ATR 趋势护城河】: 收盘价必须高于 MA20 加上 1.5 倍的 ATR，说明彻底脱离了底部泥潭
        condition_j = latest['收盘'] > (latest['MA20'] + 1.5 * latest['ATR_14'])
        
        # 【绝对底线】必须全部满足：均线达标 + 量价齐升 + 资金净流入（OBV）
        cond_base = condition_a and condition_b and condition_h

        # 【弹性加分项】计算共振得分 (0-5分)
        resonance_score = 0
        if condition_e:
            resonance_score += 1 # MACD
        if condition_f:
            resonance_score += 1 # KDJ
        if condition_g:
            resonance_score += 1 # BOLL
        if condition_i:
            resonance_score += 1 # VCP 收缩突破
        if condition_j:
            resonance_score += 1 # ATR 趋势确立

        # 新的通过条件：满足趋势底线，且五大指标至少有2个共振
        all_conditions_met = cond_base and (resonance_score >= 2)

        if all_conditions_met:
            qualified_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '最新价': latest['收盘'],
                '开盘价': latest['开盘'],
                '涨跌幅': stock.涨跌幅 if hasattr(stock, '涨跌幅') else 0,
                '成交量': latest['成交量'],
                '换手率': stock.换手率 if hasattr(stock, '换手率') else 0,
                '成交额': stock.成交额 if hasattr(stock, '成交额') else 0,
                '流通市值': stock.流通市值 if hasattr(stock, '流通市值') else 0,
                'MA5': latest['MA5'],
                'MA10': latest['MA10'],
                'MA20': latest['MA20'],
                '20日最高': latest['20日最高'],
                '均线多头': condition_a,
                '量价齐升': condition_b,
                '有效突破': condition_c,
                '拒绝追高': condition_d,
                'MACD金叉': condition_e,
                'KDJ金叉': condition_f,
                'BOLL触轨': condition_g,
                'VCP收缩突破': condition_i,
                'ATR趋势确立': condition_j,
                '共振星级': resonance_score,
            })
        else:
            failed_reasons = []
            if not condition_a:
                failed_reasons.append('均线多头不符')
            if not condition_b:
                failed_reasons.append('量价未齐升')
            if not condition_h:
                failed_reasons.append('资金净流入不符')
            if cond_base and resonance_score < 2:
                failed_reasons.append('无高级指标共振')
            if not condition_c:
                failed_reasons.append('未有效突破')
            if not condition_d:
                failed_reasons.append('距MA5过高')

            failed_stocks.append({
                '代码': stock_code,
                '名称': stock_name,
                '失败原因': ', '.join(failed_reasons)
            })

        time.sleep(0.1)

    print(f"=== 第二阶段筛选完成 ===")
    print(f"通过二筛: {len(qualified_stocks)} 只")
    print(f"未通过: {len(failed_stocks)} 只")

    return qualified_stocks, failed_stocks


def get_ai_analysis(stock, phase_1_df):
    """使用 AI 进行深度分析"""
    if not API_KEY:
        return None

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)

        # 从 phase_1_df 获取板块信息
        sector_info = "未知"
        if not phase_1_df.empty:
            stock_row = phase_1_df[phase_1_df['代码'] == stock['代码']]
            if not stock_row.empty and '所属板块' in stock_row.columns:
                sector_info = stock_row.iloc[0]['所属板块']

        current_price = float(stock['最新价'])
        ma5 = float(stock['MA5'])
        open_price = float(stock['开盘价'])
        high_20d = float(stock['20日最高'])
        atr_14 = float(stock['ATR_14'])
        
        base_support = max(ma5, open_price)
        defense_line = round(base_support - 1.2 * atr_14, 2)
        max_loss_price = round(current_price * 0.92, 2)
        defense_line = max(defense_line, max_loss_price)
        # 追求高赔率
        risk_amount = current_price - defense_line
        rr_target = current_price + (risk_amount * 1.5)
        momentum_target = high_20d + (0.5 * atr_14) 
        target_line = round(max(rr_target, momentum_target), 2)

        # 动态构建技术指标描述字符串
        indicators_str = []
        if stock.get('MACD金叉'):
            indicators_str.append("MACD金叉")
        if stock.get('KDJ金叉'):
            indicators_str.append("KDJ多头向上")
        if stock.get('BOLL触轨'):
            indicators_str.append("BOLL上轨突破")
        if stock.get('VCP收缩突破'):
            indicators_str.append('VCP收缩突破')
        if stock.get('ATR趋势确立'):
            indicators_str.append('ATR趋势确立')
        tech_desc = "、".join(indicators_str) if indicators_str else "基础量价异动"

        today_str = datetime.now().strftime("%Y年%m月%d日")
        ai_prompt = f"""今天是 {today_str}。你是资深A股游资。
今日尾盘捕获【{stock['名称']} ({stock['代码']})】，属【{sector_info}】板块。
现价 {current_price:.2f}，技术面 {stock.get('共振星级', 1)} 星级共振。
绝对防守线: {defense_line:.2f}，阻力突破线: {target_line:.2f}。
请简短分析：1.上涨原因与未来展望？2.突破确定性？3.近期有无减持/暴雷风险？
【警告】请注意当前年份是 {today_str[:4]} 年。必须严格控制在 250 字以内输出核心结论！"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": ai_prompt}],
            temperature=0.7,
            timeout = 15.0
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"AI 分析失败 ({stock['代码']}): {e}")
        return None


def main():
    """主执行函数"""
    print("=" * 60)
    print("A股选股与异动监控 自动化脚本")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 步骤1: 获取实时数据
    print("\n[步骤1] 获取全市场实时数据...")
    stock_data = get_realtime_data()

    if stock_data is None or stock_data.empty:
        print("无法获取数据，脚本终止")
        return

    print(f"成功获取 {len(stock_data)} 只股票数据")

    # 步骤2: 执行一筛
    print("\n[步骤2] 执行第一阶段筛选...")
    phase_1_result = screen_phase_1(stock_data)

    if phase_1_result.empty:
        print("第一阶段未找到符合条件的股票，脚本终止")
        return

    print(f"第一阶段通过: {len(phase_1_result)} 只")

    # 步骤2.5: 计算板块共振
    print("\n[步骤2.5] 计算板块共振...")
    sector_resonance = calculate_sector_resonance(phase_1_result)
    if sector_resonance:
        sector_str_list = [f"{name}({count}只)" for name, count in sector_resonance.items()]
        sector_str = '、'.join(sector_str_list)
        print(f"今日资金攻击主线: {sector_str}")
    else:
        sector_str = ""
        print("无板块共振")

    # 步骤3: 执行二筛
    print("\n[步骤3] 执行第二阶段技术形态筛选...")
    qualified_stocks, failed_stocks = screen_phase_2(phase_1_result)

    if not qualified_stocks:
        print("第二阶段未找到符合条件的标的，今日无推荐")
        return

    print(f"第二阶段通过: {len(qualified_stocks)} 只")

    # 步骤4: 按共振星级降序排列
    print("\n[步骤4] 按共振星级降序排列...")
    qualified_stocks = sorted(qualified_stocks, key=lambda x: x['共振星级'], reverse=True)

    # 步骤5: AI 批量研判与拼装推送文案（化整为零，单发推送）
    print("\n[步骤5] 生成研报并逐个推送...")

    phase_1_df = phase_1_result  # 保存一筛结果供 AI 分析使用

    # 先发总览
    overview_title = f"🎯 尾盘监控完成"
    overview_body = f"共发现 {len(qualified_stocks)} 只标的，正在生成研报..."
    if sector_str:
        overview_body += f"\n今日板块共振: {sector_str}"
    send_bark(overview_title, overview_body)
    time.sleep(1.5)

    # 遍历股票，逐个发送推送
    for i, stock in enumerate(qualified_stocks, 1):
        
        current_price = float(stock['最新价'])
        ma5 = float(stock['MA5'])
        open_price = float(stock['开盘价'])
        high_20d = float(stock['20日最高'])
        atr_14 = float(stock['ATR_14'])
        # 动态止损线（8%保底）
        base_support = max(ma5, open_price)
        defense_line = round(base_support - 1.2 * atr_14, 2)
        max_loss_price = round(current_price * 0.92, 2)
        defense_line = max(defense_line, max_loss_price)
        # 追求高赔率
        risk_amount = current_price - defense_line
        rr_target = current_price + (risk_amount * 1.5)
        momentum_target = high_20d + (0.5 * atr_14) 
        target_line = round(max(rr_target, momentum_target), 2)

        # 获取板块信息
        sector_info = "未知"
        if not phase_1_df.empty:
            stock_row = phase_1_df[phase_1_df['代码'] == stock['代码']]
            if not stock_row.empty and '所属板块' in stock_row.columns:
                sector_info = stock_row.iloc[0]['所属板块']

        # 获取换手率
        turnover = stock.get('换手率', 0)

        # 构建股票信息
        star_display = "⭐" * stock['共振星级']
        stock_body = (
            f"现价: {current_price:.2f} | 涨跌幅: {stock['涨跌幅']:.2f}%\n"
            f"板块: {sector_info} | 换手率: {turnover:.2f}%\n"
            f"共振星级: {stock['共振星级']}{star_display}\n"
            f"绝对防守线: {defense_line:.2f}\n"
            f"阻力突破线: {target_line:.2f}"
        )

        # 仅对 共振星级 >= 4 的股票触发 AI 分析
        if stock['共振星级'] >= 4:
            print(f"正在分析 {stock['名称']} (共振星级 {stock['共振星级']}⭐)...")
            ai_result = get_ai_analysis(stock, phase_1_df)
            if ai_result:
                stock_body += f"\n\n🤖 AI研判: {ai_result}"

        # 发送单只股票推送
        push_title = f"[{i}/{len(qualified_stocks)}] {stock['名称']} ({stock['代码']})"
        send_bark(push_title, stock_body)
        time.sleep(1.5)  # 防止并发过快被 Bark 拦截

    print("\n脚本执行完成！")


if __name__ == "__main__":
    main()
