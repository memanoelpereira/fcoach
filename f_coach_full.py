#!/usr/bin/env python3
"""
F-Coach Full V8.2 · 9.9 Bayes contextual + Conformal + Live ML — Market Coach em Streamlit
US30 · US500 · US100 · EUR/USD · USD/CHF · EUR/CHF · Forex · Commodities

Versão dirigida: engenharia elaborada por baixo, interface simples por cima, com estratégia calibrada por janela operacional e modo dirigido por meta de capital/lucro.

Princípios da V6:
1) Tela inicial minimalista: decisão, motivo, risco e plano de operação.
2) Núcleo técnico robusto: indicadores, regime, volatilidade, suporte/resistência e sessão.
3) ML ensemble temporal: múltiplos modelos, walk-forward, confiança encolhida e qualidade mínima.
4) Gates adaptativos: bloqueiam ou reforçam sinais por contexto histórico fora da amostra.
5) Backtest operacional: stop/alvo em ATR, custo, slippage, drawdown e profit factor.
6) Plano de risco: stop/target sugeridos, relação risco-retorno e tamanho por risco fixo.
7) Copiloto IA na aba simples: diálogo contextual durante aceite/recusa dos sinais.
8) Registro operacional em tempo real: aceite/recusa, posição aberta, marcação manual e P&L realizado/não realizado.
9) Observabilidade: auditoria do score, saúde dos dados, diário e exportação.
10) Camada temporal: estratégias, ML, stop/alvo e risco calibrados para janelas de 1, 5, 15, 30, 60, 120 e 240 minutos.
11) Modo dirigido: o operador define capital alocado e meta de lucro; o app gera roteiros condicionais, risco por operação e acompanhamento da meta.

Execução local:
    streamlit run f_coach_full_v8_1_98.py

Notas:
- O app não executa ordens, não é recomendação financeira e não substitui gestão de risco.
- A camada de dados usa fontes públicas (Yahoo/yfinance), que podem ter atraso, falhas ou cobertura irregular.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Literal, Optional, Any
import hashlib
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:  # gráficos melhores quando disponível; fallback para st.line_chart
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import permutation_importance
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    HistGradientBoostingClassifier = None
    ExtraTreesClassifier = None
    VotingClassifier = None
    LogisticRegression = None
    balanced_accuracy_score = None
    f1_score = None
    matthews_corrcoef = None
    TimeSeriesSplit = None
    StandardScaler = None
    Pipeline = None
    permutation_importance = None

try:
    import feedparser
except Exception:  # pragma: no cover
    feedparser = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

APP_VERSION = "8.5-forex-volume-safe"
ET = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("America/Maceio")

TICKER_MAP = {
    "US30": "YM=F", "US500": "ES=F", "US100": "NQ=F",
    "US30_CASH": "^DJI", "US500_CASH": "^GSPC", "US100_CASH": "^NDX",
    "VIX": "^VIX", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "US2Y": "^IRX",
    "WTI": "CL=F", "BRENT": "BZ=F", "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X", "USDCHF": "USDCHF=X", "EURCHF": "EURCHF=X",
    "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X",
}

ASSET_TYPE = {
    "US30": "index", "US500": "index", "US100": "index",
    "US30_CASH": "index", "US500_CASH": "index", "US100_CASH": "index",
    "VIX": "volatility", "DXY": "macro", "US10Y": "rates", "US2Y": "rates",
    "WTI": "commodity", "BRENT": "commodity", "XAUUSD": "commodity",
    "EURUSD": "forex", "USDCHF": "forex", "EURCHF": "forex",
    "GBPUSD": "forex", "USDJPY": "forex",
    "AUDUSD": "forex", "USDCAD": "forex", "NZDUSD": "forex", "EURGBP": "forex",
}

STABLE_FOREX_SYMBOLS = ["EURUSD", "USDCHF", "EURCHF"]
DEFAULT_INDEX_SYMBOLS = ["US30", "US500", "US100"]
DEFAULT_SYMBOLS = DEFAULT_INDEX_SYMBOLS + STABLE_FOREX_SYMBOLS
DEFAULT_STARTUP_SYMBOLS = ["US500", "EURUSD"]  # abertura leve; demais ativos podem ser marcados depois
CONTEXT_SYMBOLS = ["VIX", "DXY", "US10Y", "WTI", "XAUUSD"]

FOREX_STABILITY_TIER = {
    "EURUSD": "estável",
    "USDCHF": "estável",
    "EURCHF": "estável",
    "USDJPY": "intermediário",
    "AUDUSD": "intermediário",
    "USDCAD": "intermediário",
    "NZDUSD": "intermediário",
    "GBPUSD": "volátil",
    "EURGBP": "intermediário",
}

FOREX_TIER_PROFILE = {
    "estável": {
        "bias": "priorizar confirmação, pullback e reversão controlada",
        "min_rr_hint": 1.15,
        "note": "Par de maior estabilidade relativa: bom para elevar acerto, desde que spread/dados estejam OK.",
    },
    "intermediário": {
        "bias": "exigir confirmação macro ou força direcional",
        "min_rr_hint": 1.30,
        "note": "Par útil quando o contexto macro está claro; evitar sinais isolados.",
    },
    "volátil": {
        "bias": "usar apenas com alta convicção e stop mais largo",
        "min_rr_hint": 1.60,
        "note": "Par mais agressivo; tende a reduzir acerto bruto se operado sem filtro forte.",
    },
}


DATA_DIR = Path(os.getenv("F_COACH_DATA_DIR", "/mnt/data" if Path("/mnt/data").exists() else "."))
JOURNAL_PATH = DATA_DIR / "f_coach_journal_v7.csv"
TRADE_LOG_PATH = DATA_DIR / "f_coach_trades_v7.csv"
DECISION_LOG_PATH = DATA_DIR / "f_coach_decisions_v7.csv"
ALERTS_SENT_PATH = DATA_DIR / "f_coach_alerts_sent_v7.json"
GATES_CACHE_PATH = DATA_DIR / "f_coach_context_gates_v81.json"

TELEGRAM_TOKEN = os.getenv("F_COACH_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("F_COACH_TELEGRAM_CHAT_ID", "")

Action = Literal["COMPRAR", "VENDER", "AGUARDAR", "SAIR · PROTEGER"]

# Critérios prudenciais globais
HIGH_CONVICTION_MIN_SCORE = 1.55
HIGH_CONVICTION_MIN_ML_QUALITY = 0.52
HIGH_CONVICTION_MIN_ML_PROB = 0.63
HIGH_CONVICTION_GOOD_SESSIONS = {"MANHÃ", "TARDE"}

# Gates aprendidos
MIN_CONTEXT_OPS = 10
POOR_CONTEXT_RET_PCT = -0.25
POOR_CONTEXT_HIT = 0.46
GOOD_CONTEXT_RET_PCT = 0.18
GOOD_CONTEXT_HIT = 0.54

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketSession:
    name: str
    style: str
    note: str
    liquidity: str
    directional_permission: float

@dataclass
class DataHealth:
    symbol: str
    rows: int
    last_time: str
    minutes_stale: float
    status: str
    note: str

@dataclass
class CoachReport:
    symbol: str
    asset_type: str
    price: float
    change_pct: float
    rsi: float
    macd_hist: float
    stoch_k: float
    stoch_d: float
    atr: float
    atr_pct: float
    bb_pct: float
    vol_ratio: float
    ema9: float
    ema21: float
    ema50: float
    ema200: float
    vwap_val: float
    support: float
    resistance: float
    action: Action
    urgency: str
    trend: str
    trend_strength: str
    reversal_risk: str
    session: MarketSession
    score_bull: int
    score_bear: int
    confidence_rule: float
    data_health: DataHealth
    messages: list[str] = field(default_factory=list)

@dataclass
class MLReport:
    symbol: str
    available: bool
    prediction: str = "INDISPONÍVEL"
    probability: float = np.nan
    calibrated_probability: float = np.nan
    model_quality: float = np.nan
    f1_macro: float = np.nan
    mcc: float = np.nan
    horizon_bars: int = 3
    class_balance: dict[str, int] = field(default_factory=dict)
    model_table: list[dict[str, Any]] = field(default_factory=list)
    top_features: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    # Conformal prediction (Adaptive Prediction Sets) — incerteza com cobertura
    # empírica validada, não um proxy heurístico. Ver _conformal_calibrate().
    conformal_available: bool = False
    conformal_alpha: float = 0.10                    # 1 - cobertura nominal alvo
    conformal_set: list[str] = field(default_factory=list)   # classes no conjunto
    conformal_coverage_empirical: float = np.nan      # cobertura medida no holdout
    conformal_set_size_avg: float = np.nan            # tamanho médio histórico do conjunto
    conformal_note: str = ""

@dataclass
class AgentOpinion:
    name: str
    vote: str
    confidence: float
    reason: str

@dataclass
class ContextReport:
    risk_regime: str
    risk_score: float
    macro_notes: list[str]
    news_items: list[dict]
    news_pressure: str
    news_score: float
    cross_asset_notes: list[str] = field(default_factory=list)

@dataclass
class DecisionAudit:
    raw_decision: str
    raw_score: float
    final_decision: str
    final_score: float
    margin: float
    blockers: list[str]
    boosts: list[str]
    components: dict[str, float]
    rationale: str

@dataclass
class RiskPlan:
    symbol: str
    decision: str
    entry: float
    stop: float
    target: float
    risk_points: float
    reward_points: float
    rr: float
    suggested_units: float
    risk_cash: float
    invalidation: str

@dataclass(frozen=True)
class TimeframeStrategy:
    minutes: int
    label: str
    data_interval: str
    default_period: str
    horizon_bars: int
    stop_atr: float
    target_atr: float
    risk_factor: float
    min_session_permission: float
    style: str
    preferred_setups: str
    confirmations: str
    avoid: str
    management: str

@dataclass
class DirectedGoal:
    enabled: bool
    capital_cash: float
    target_profit_cash: float
    max_loss_cash: float
    max_trades: int
    point_value: float
    min_rr: float
    max_units: float
    allow_partial_targets: bool

@dataclass
class DirectedRoute:
    symbol: str
    action: str
    priority: int
    route_score: float
    entry: float
    stop: float
    target: float
    units: float
    risk_cash: float
    reward_cash: float
    rr: float
    target_coverage: float
    status: str
    roteiro: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Camada temporal: estratégia por janela operacional
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_WINDOWS = [1, 5, 15, 30, 60, 120, 240]


def timeframe_strategy(minutes: int) -> TimeframeStrategy:
    """Configuração prudencial por janela de operação.

    A janela temporal muda o problema: um trade de 1 minuto deve ser tratado
    como micro-scalp, enquanto 120/240 minutos pedem tese de regime e gestão
    mais paciente. Esta função centraliza intervalo dos dados, horizonte do ML,
    stop/alvo em ATR, fração de risco e tipo de confirmação exigida.
    """
    m = int(minutes)
    strategies = {
        1: TimeframeStrategy(
            1, "1 min · micro-scalp", "1m", "7d", 1,
            0.28, 0.42, 0.25, 0.85,
            "Micro-scalp defensivo",
            "rejeição curta em VWAP/EMA9, rompimento de microfaixa ou pullback muito rápido",
            "spread baixo, candle fechado a favor, volume/VRAT acima da média e nenhum bloqueio de sessão",
            "notícias, abertura desordenada, dados atrasados, candle muito esticado ou P&L emocional",
            "alvo curto, stop rígido, saída parcial rápida; se não andar quase imediatamente, zerar ou proteger",
        ),
        5: TimeframeStrategy(
            5, "5 min · scalp confirmado", "1m", "7d", 5,
            0.38, 0.70, 0.35, 0.70,
            "Scalp com confirmação",
            "pullback em EMA9/EMA21, retorno ao VWAP, rompimento com reteste curto",
            "concordância técnica+sessão; ML não pode discordar forte; evitar operar contra força relativa clara",
            "laterais estreitas, almoço NY, overnight sem liquidez, reversão alta contra a entrada",
            "mover stop para proteção após avanço de ~0,5 ATR; não transformar scalp em trade longo",
        ),
        15: TimeframeStrategy(
            15, "15 min · momentum intraday", "5m", "30d", 3,
            0.55, 1.00, 0.55, 0.60,
            "Momentum intraday",
            "continuidade após consolidação, reteste de rompimento, reversão em suporte/resistência com confluência",
            "tendência ao menos moderada, RSI sem exaustão, MACD coerente e sessão não bloqueada",
            "comprar resistência sem reteste, vender suporte sem confirmação, volatilidade extrema pós-notícia",
            "stop técnico por ATR; alvo mínimo próximo de 1:1,8; aceitar tempo de maturação de poucos candles",
        ),
        30: TimeframeStrategy(
            30, "30 min · trade tático", "5m", "60d", 6,
            0.70, 1.30, 0.70, 0.55,
            "Trade tático intraday",
            "tendência com pullback, reversão de faixa maior, rompimento validado por força relativa",
            "duas camadas a favor: técnica+ML, técnica+macro ou força relativa+sessão",
            "sinal misto, macro defensivo contra compra em índice, perda rápida de VWAP/EMA21",
            "permitir oscilação moderada; reduzir tamanho se ATR estiver alto; revisar no meio da janela",
        ),
        60: TimeframeStrategy(
            60, "60 min · tendência curta", "15m", "60d", 4,
            0.85, 1.60, 0.85, 0.50,
            "Tendência curta intraday",
            "continuidade de tendência, pullback em EMA21/EMA50, reversão confirmada em zona relevante",
            "regime técnico claro, risco de reversão baixo/médio, contexto macro não hostil à direção",
            "lateralidade, divergência forte do ML, entrada na metade final de movimento já exausto",
            "stop mais folgado; alvo por zonas; aceitar ruído, mas sair se a tese de tendência falhar",
        ),
        120: TimeframeStrategy(
            120, "120 min · swing intraday", "30m", "60d", 4,
            1.05, 2.00, 1.00, 0.45,
            "Swing intraday conservador",
            "tese de regime, continuação após consolidação ampla, reversão apenas com sinal forte",
            "tendência moderada/forte, força relativa favorável, macro neutro/construtivo para compra em índice",
            "scalps emocionais, sinais contra tendência, operar notícia sem digestão do mercado",
            "planejar antes; stop estrutural; parcial em 1 ATR; reavaliar se sessão mudar de caráter",
        ),
        240: TimeframeStrategy(
            240, "240 min · posição curta", "1h", "90d", 4,
            1.25, 2.40, 1.10, 0.40,
            "Posição curta / swing de sessão",
            "tese de direção dominante, alinhamento de EMAs, macro e força relativa; entrada preferida em pullback",
            "confirmação multicomponente: técnica + macro/sessão + ML sem oposição relevante",
            "entrar por impulso em candle esticado, ignorar eventos macro, manter tese após invalidação técnica",
            "gestão paciente; stop estrutural; alvo por resistência/suporte maior; revisar a cada candle de 1h",
        ),
    }
    return strategies.get(m, strategies[60])


def timeframe_label(minutes: int) -> str:
    return timeframe_strategy(int(minutes)).label


def timeframe_overview(strategy: TimeframeStrategy) -> dict[str, Any]:
    return {
        "janela": strategy.label,
        "intervalo_dados": strategy.data_interval,
        "histórico": strategy.default_period,
        "horizonte_ML_barras": strategy.horizon_bars,
        "stop_ATR": strategy.stop_atr,
        "alvo_ATR": strategy.target_atr,
        "risco_relativo": strategy.risk_factor,
        "estilo": strategy.style,
    }


def timeframe_strategy_markdown(strategy: TimeframeStrategy) -> str:
    return (
        f"**Estratégia da janela — {strategy.label}.** "
        f"Estilo: {strategy.style}. Setup preferencial: {strategy.preferred_setups}. "
        f"Confirmações: {strategy.confirmations}. Evitar: {strategy.avoid}. "
        f"Gestão: {strategy.management}."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Dados e indicadores
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def fetch(symbol: str, period: str = "90d", interval: str = "1h") -> pd.DataFrame:
    ticker = TICKER_MAP.get(symbol.upper(), symbol.upper())
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    out = df[needed].dropna().copy()
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="last")]
    return out

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(close, 12) - ema(close, 26)
    sig = ema(m, 9)
    return m, sig, m - sig

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def bollinger(close: pd.Series, n: int = 20, k: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = close.rolling(n).mean()
    s = close.rolling(n).std()
    return m + k * s, m, m - k * s

def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    pct_k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP robusto a Forex/CFDs sem volume real.

    Em muitos pares de moedas baixados via Yahoo/yfinance, `volume` vem zerado
    ou ausente. A versão anterior transformava volume zero em NaN e, por efeito
    cascata, `vwap`, `vrat` e `vol_z` ficavam todos NaN. Como `analyze()` usa
    `dropna()`, pares como EURUSD podiam desaparecer e gerar:

        Dados insuficientes após indicadores.

    Quando não há volume útil, usamos uma média típica curta como proxy neutro
    de VWAP. Isso não finge volume; apenas impede que a ausência de volume mate
    toda a série.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol_raw = df.get("volume", pd.Series(index=df.index, dtype=float)).astype(float)
    vol = vol_raw.replace(0, np.nan)
    if vol.notna().sum() < max(5, min(20, len(df) // 10)):
        return typical.rolling(20, min_periods=1).mean()
    out = (typical * vol).cumsum() / vol.cumsum()
    return out.fillna(typical.rolling(20, min_periods=1).mean())

def rolling_z(s: pd.Series, n: int = 50) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std().replace(0, np.nan)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi"] = rsi(d["close"])
    d["macd"], d["macd_sig"], d["macd_hist"] = macd(d["close"])
    d["atr"] = atr(d)
    d["bb_u"], d["bb_m"], d["bb_l"] = bollinger(d["close"])
    for n in [9, 21, 50, 100, 200]:
        d[f"ema{n}"] = ema(d["close"], n)
    d["sk"], d["sd"] = stochastic(d)
    d["vwap"] = vwap(d)
    # Forex/CFDs frequentemente vêm com volume zero no Yahoo. Nesses casos,
    # razão de volume e z-score de volume devem ser neutros, não NaN.
    vol_mean = d["volume"].replace(0, np.nan).rolling(20, min_periods=5).mean()
    d["vrat"] = (d["volume"].replace(0, np.nan) / vol_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    d["ret_1"] = d["close"].pct_change()
    d["ret_3"] = d["close"].pct_change(3)
    d["ret_6"] = d["close"].pct_change(6)
    d["ret_12"] = d["close"].pct_change(12)
    d["range_pct"] = (d["high"] - d["low"]) / d["close"]
    d["body_pct"] = (d["close"] - d["open"]) / d["close"]
    d["upper_wick_pct"] = (d["high"] - d[["open", "close"]].max(axis=1)) / d["close"]
    d["lower_wick_pct"] = (d[["open", "close"]].min(axis=1) - d["low"]) / d["close"]
    d["vol_z"] = rolling_z(d["volume"].replace(0, np.nan), 50).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["atr_z"] = rolling_z(d["atr"], 50).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Garantia extra para pares FX: VWAP/volume não podem eliminar toda a série.
    d["vwap"] = d["vwap"].replace([np.inf, -np.inf], np.nan).fillna(d["close"].rolling(20, min_periods=1).mean())
    d["vrat"] = d["vrat"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return d

def data_health(symbol: str, df: pd.DataFrame, interval: str) -> DataHealth:
    if df.empty:
        return DataHealth(symbol, 0, "", np.inf, "RUIM", "Sem dados retornados.")
    last = df.index[-1]
    if isinstance(last, pd.Timestamp):
        if last.tzinfo is None:
            last_dt = last.tz_localize("UTC").tz_convert(LOCAL_TZ).to_pydatetime()
        else:
            last_dt = last.tz_convert(LOCAL_TZ).to_pydatetime()
    else:
        last_dt = datetime.now(LOCAL_TZ)
    minutes = (datetime.now(LOCAL_TZ) - last_dt).total_seconds() / 60
    expected = {"1m": 8, "2m": 12, "5m": 25, "15m": 60, "30m": 90, "1h": 180, "60m": 180, "1d": 60 * 36}.get(interval, 180)
    if len(df) < 80:
        return DataHealth(symbol, len(df), last_dt.isoformat(timespec="minutes"), minutes, "FRÁGIL", "Histórico curto para indicadores/ML.")
    if minutes > expected:
        return DataHealth(symbol, len(df), last_dt.isoformat(timespec="minutes"), minutes, "ATRASADO", "Último candle parece antigo para o intervalo escolhido.")
    return DataHealth(symbol, len(df), last_dt.isoformat(timespec="minutes"), minutes, "OK", "Dados recentes o suficiente para leitura operacional.")

# ─────────────────────────────────────────────────────────────────────────────
# Sessão e contexto
# ─────────────────────────────────────────────────────────────────────────────

def market_session(now: Optional[datetime] = None) -> MarketSession:
    now = now or datetime.now(LOCAL_TZ)
    now_local = now.replace(tzinfo=LOCAL_TZ) if now.tzinfo is None else now.astimezone(LOCAL_TZ)
    wd = now_local.weekday()
    t_local = now_local.hour * 60 + now_local.minute

    if wd == 4 and t_local >= 18 * 60:
        return MarketSession("FECHADO (FDS FUTUROS)", "gray", "Futuros fechados para o fim de semana; usar apenas revisão.", "zero", 0.0)
    if wd == 5:
        return MarketSession("FECHADO (FDS FUTUROS)", "gray", "Sábado: futuros fechados; ótimo para backtest e diário.", "zero", 0.0)
    if wd == 6 and t_local < 19 * 60:
        return MarketSession("FECHADO (FDS FUTUROS)", "gray", "Domingo antes das 19h de Brasília: futuros ainda fechados.", "zero", 0.0)
    if 18 * 60 <= t_local < 19 * 60:
        return MarketSession("PAUSA FUTUROS", "gray", "Pausa diária dos futuros; não gerar entrada nova.", "zero", 0.0)

    now_et = now_local.astimezone(ET)
    t_et = now_et.hour * 60 + now_et.minute
    if t_et < 4 * 60:
        return MarketSession("OVERNIGHT FUTUROS", "blue", "Overnight: liquidez menor; sinal precisa de confirmação extra.", "baixa", 0.45)
    if t_et < 9 * 60 + 30:
        return MarketSession("PRÉ-MARKET FUTUROS", "orange", "Pré-market: bom para leitura de gap, ruim para impulso sem confirmação.", "média", 0.62)
    if t_et < 10 * 60:
        return MarketSession("ABERTURA", "red", "Abertura cash: volatilidade elevada; reduzir tamanho e aguardar estabilização.", "alta/instável", 0.55)
    if t_et < 12 * 60:
        return MarketSession("MANHÃ", "green", "Janela favorável para tendência e continuidade.", "alta", 1.0)
    if t_et < 14 * 60:
        return MarketSession("ALMOÇO NY", "orange", "Volume tende a cair; evitar rompimentos fracos.", "baixa/média", 0.55)
    if t_et < 15 * 60 + 30:
        return MarketSession("TARDE", "green", "Segunda janela operacional; observar continuidade/reversão do fluxo.", "alta", 0.95)
    if t_et < 16 * 60:
        return MarketSession("PRÉ-FECHAMENTO", "orange", "Últimos 30 minutos do cash: ajustes e realização podem distorcer sinais.", "alta/instável", 0.50)
    return MarketSession("PÓS-MERCADO FUTUROS", "blue", "Cash fechado; futuros abertos com leitura conservadora.", "baixa/média", 0.42)

def index_market_closed_or_paused_for(symbol: str, sess: MarketSession) -> bool:
    """Bloqueio operacional duro para índices/futuros.

    Quando os futuros estão fechados ou em pausa, yfinance normalmente devolve
    o último candle disponível. Esses dados são úteis para revisão, mas NÃO
    podem gerar coaching ao vivo, alerta Telegram ou SAIR/PROTEGER automático.
    """
    return ASSET_TYPE.get(str(symbol).upper(), "") == "index" and float(sess.directional_permission) <= 0.0

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_news(max_items: int = 10) -> list[dict]:
    items: list[dict] = []
    rss_urls = ["https://finance.yahoo.com/news/rssindex", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"]
    if feedparser is not None:
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in getattr(feed, "entries", [])[:max_items]:
                    title = str(getattr(entry, "title", "")).strip()
                    link = str(getattr(entry, "link", "")).strip()
                    published = str(getattr(entry, "published", "")).strip()
                    if title:
                        items.append({"title": title, "link": link, "published": published, "source": url})
                    if len(items) >= max_items:
                        break
            except Exception:
                continue
            if len(items) >= max_items:
                break
    if not items:
        try:
            for n in (getattr(yf.Ticker("^GSPC"), "news", None) or [])[:max_items]:
                title = str(n.get("title", "")).strip()
                if title:
                    items.append({"title": title, "link": n.get("link", ""), "published": str(n.get("providerPublishTime", "")), "source": "yfinance"})
        except Exception:
            pass
    seen, deduped = set(), []
    for it in items:
        key = it["title"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(it)
    return deduped[:max_items]

def score_news(items: list[dict]) -> tuple[str, float]:
    if not items:
        return "NEUTRO", 0.0
    text = " ".join(it.get("title", "") for it in items).lower()
    neg = ["inflation", "cpi", "ppi", "tariff", "war", "geopolitical", "selloff", "recession", "default", "spike", "hawkish", "yield", "crash"]
    pos = ["soft landing", "rate cut", "cooling", "rally", "earnings beat", "risk-on", "growth", "stimulus", "record high", "optimism"]
    score = sum(1 for kw in pos if kw in text) - sum(1 for kw in neg if kw in text)
    if score <= -2:
        return "RISK-OFF", -0.70
    if score >= 2:
        return "RISK-ON", 0.55
    return "NEUTRO", 0.0

def _last_close_change(symbol: str, period: str = "10d", interval: str = "1d") -> tuple[float, float]:
    df = fetch(symbol, period=period, interval=interval)
    if df.empty or len(df) < 2:
        return np.nan, np.nan
    price = float(df["close"].iloc[-1])
    change = float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100)
    return price, change

def build_context_report(include_news: bool = True) -> ContextReport:
    vix, vix_chg = _last_close_change("VIX")
    dxy, dxy_chg = _last_close_change("DXY")
    y10, y10_chg = _last_close_change("US10Y")
    wti, wti_chg = _last_close_change("WTI")
    xau, xau_chg = _last_close_change("XAUUSD")
    notes, cross = [], []
    risk = 0.0
    if np.isfinite(vix):
        if vix >= 25:
            risk += 1.35; notes.append(f"VIX elevado ({vix:.1f}) — regime defensivo.")
        elif vix >= 20:
            risk += 0.75; notes.append(f"VIX moderadamente alto ({vix:.1f}) — risco aumentado.")
        elif vix < 14:
            risk -= 0.45; notes.append(f"VIX baixo ({vix:.1f}) — ambiente mais favorável a risco.")
        if np.isfinite(vix_chg) and vix_chg > 6:
            risk += 0.45; cross.append(f"VIX subiu {vix_chg:+.1f}%: proteger compras frágeis.")
    if np.isfinite(dxy_chg) and dxy_chg > 0.45:
        risk += 0.30; cross.append(f"DXY forte ({dxy_chg:+.2f}%): pressão potencial sobre risco.")
    if np.isfinite(y10_chg) and y10_chg > 1.25:
        risk += 0.35; cross.append(f"Yield 10Y em alta ({y10_chg:+.2f}%): atenção especial ao US100.")
    if np.isfinite(wti_chg) and abs(wti_chg) > 2.0:
        cross.append(f"WTI variou {wti_chg:+.2f}%: pode afetar inflação/energia.")
    if np.isfinite(xau_chg) and xau_chg > 1.0 and np.isfinite(vix_chg) and vix_chg > 0:
        risk += 0.15; cross.append(f"Ouro e VIX em alta: leitura defensiva moderada.")

    regime = "DEFENSIVO" if risk >= 1.15 else "CONSTRUTIVO" if risk <= -0.30 else "NEUTRO"
    news_items = fetch_market_news() if include_news else []
    news_pressure, news_score = score_news(news_items)
    return ContextReport(regime, float(risk), notes, news_items, news_pressure, float(news_score), cross)

# ─────────────────────────────────────────────────────────────────────────────
# Análise técnica e agentes
# ─────────────────────────────────────────────────────────────────────────────

def _sr(df: pd.DataFrame, n: int = 60) -> tuple[float, float]:
    r = df.tail(n)
    return float(r["low"].min()), float(r["high"].max())

def analyze(df: pd.DataFrame, symbol: str, interval: str = "1h") -> CoachReport:
    atype = ASSET_TYPE.get(symbol, "forex")
    sess = market_session()
    health = data_health(symbol, df, interval)
    if index_market_closed_or_paused_for(symbol, sess):
        # Se o mercado está fechado, o dado retornado é por definição retrospectivo.
        # Não chamar isso de sinal atrasado nem permitir que gere decisão operacional.
        health = DataHealth(symbol, len(df), health.last_time, health.minutes_stale, "FECHADO", "Mercado/futuros fechados ou em pausa; usando apenas o último candle disponível para revisão.")
    d = add_indicators(df).dropna().copy()
    if len(d) < 60:
        raise ValueError("Dados insuficientes após indicadores.")
    r, prev = d.iloc[-1], d.iloc[-2]
    price = float(r["close"])
    chg_pct = float((price / float(prev["close"]) - 1) * 100)
    atr_v = float(r["atr"])
    atr_pct = atr_v / price * 100 if price else np.nan
    rsi_v, sk_v, sd_v = float(r["rsi"]), float(r["sk"]), float(r["sd"])
    macd_h, macd_prev = float(r["macd_hist"]), float(prev["macd_hist"])
    vrat_v = float(r["vrat"]) if np.isfinite(r["vrat"]) else 1.0
    vwap_v = float(r["vwap"]) if np.isfinite(r["vwap"]) else price
    sup, res = _sr(d)
    bb_range = float(r["bb_u"]) - float(r["bb_l"])
    bb_pct = float((price - r["bb_l"]) / bb_range) if bb_range > 0 else 0.5

    rsi_ob, rsi_os = (75, 35) if atype == "index" else (72, 32) if atype == "commodity" else (70, 30)
    sr_tol = atr_v * (0.50 if atype == "index" else 0.35)

    ema_up = float(r["ema9"]) > float(r["ema21"]) > float(r["ema50"])
    ema_dn = float(r["ema9"]) < float(r["ema21"]) < float(r["ema50"])
    above_200 = price > float(r["ema200"])
    above_vwap = price > vwap_v
    if ema_up and above_200:
        trend, tstr = "ALTA", "FORTE"
    elif ema_up:
        trend, tstr = "ALTA", "MODERADA"
    elif ema_dn and not above_200:
        trend, tstr = "BAIXA", "FORTE"
    elif ema_dn:
        trend, tstr = "BAIXA", "MODERADA"
    else:
        trend, tstr = "LATERAL", "FRACA"

    msgs, bull, bear = [], 0, 0
    if rsi_v < rsi_os:
        msgs.append(f"RSI sobrevendido ({rsi_v:.0f}) — pode surgir defesa compradora."); bull += 2
    elif rsi_v < rsi_os + 10:
        msgs.append(f"RSI em zona baixa ({rsi_v:.0f}) — atenção para repique."); bull += 1
    elif rsi_v > rsi_ob:
        msgs.append(f"RSI sobrecomprado ({rsi_v:.0f}) — risco de realização."); bear += 2
    elif rsi_v > rsi_ob - 8:
        msgs.append(f"RSI aquecido ({rsi_v:.0f}) — momentum no limite."); bear += 1

    if macd_h > 0 and macd_prev <= 0:
        msgs.append("MACD cruzou para cima — sinal comprador."); bull += 3
    elif macd_h < 0 and macd_prev >= 0:
        msgs.append("MACD cruzou para baixo — sinal vendedor."); bear += 3
    elif macd_h > 0 and macd_h > macd_prev:
        msgs.append("MACD positivo e expandindo — momentum comprador."); bull += 1
    elif macd_h < 0 and macd_h < macd_prev:
        msgs.append("MACD negativo e caindo — pressão vendedora."); bear += 1

    if sk_v < 20 and sd_v < 20:
        msgs.append("Estocástico sobrevendido — possível repique."); bull += 1
    elif sk_v > 80 and sd_v > 80:
        msgs.append("Estocástico sobrecomprado — possível correção."); bear += 1
    if sk_v > sd_v and float(prev["sk"]) <= float(prev["sd"]) and sk_v < 50:
        msgs.append("Cruzamento estocástico de alta em zona baixa."); bull += 2
    elif sk_v < sd_v and float(prev["sk"]) >= float(prev["sd"]) and sk_v > 50:
        msgs.append("Cruzamento estocástico de baixa em zona alta."); bear += 2

    if bb_pct > 0.97:
        msgs.append("Preço colado à banda superior — breakout ou exaustão."); bear += 1
    elif bb_pct < 0.03:
        msgs.append("Preço colado à banda inferior — reversão ou queda livre."); bull += 1

    if atype == "index":
        if vrat_v > 1.5 and trend == "ALTA":
            msgs.append(f"Volume {vrat_v:.1f}x acima da média — alta com participação."); bull += 2
        elif vrat_v > 1.5 and trend == "BAIXA":
            msgs.append(f"Volume {vrat_v:.1f}x acima da média — queda com distribuição."); bear += 2
        elif vrat_v < 0.5:
            msgs.append("Volume baixo — confluência menos confiável.")
        if above_vwap and trend == "ALTA":
            msgs.append("Preço acima do VWAP em tendência de alta."); bull += 1
        elif not above_vwap and trend == "BAIXA":
            msgs.append("Preço abaixo do VWAP em tendência de baixa."); bear += 1

    if abs(price - sup) <= sr_tol:
        msgs.append(f"Preço próximo ao suporte ({sup:.2f})."); bull += 1
    if abs(price - res) <= sr_tol:
        msgs.append(f"Preço próximo à resistência ({res:.2f})."); bear += 1
    if trend == "ALTA" and tstr == "FORTE":
        msgs.append("EMAs alinhadas em alta forte."); bull += 1
    elif trend == "BAIXA" and tstr == "FORTE":
        msgs.append("EMAs alinhadas em baixa forte."); bear += 1

    rev = 0
    if (trend == "ALTA" and rsi_v > rsi_ob - 5) or (trend == "BAIXA" and rsi_v < rsi_os + 5): rev += 1
    if bb_pct > 0.9 or bb_pct < 0.1: rev += 1
    if (trend == "ALTA" and bear >= 3) or (trend == "BAIXA" and bull >= 3): rev += 1
    if atype == "index" and vrat_v > 2.0 and ((trend == "ALTA" and chg_pct < 0) or (trend == "BAIXA" and chg_pct > 0)): rev += 1
    reversal_risk = "ALTO" if rev >= 2 else "MÉDIO" if rev == 1 else "BAIXO"

    total = bull + bear
    confidence = abs(bull - bear) / total if total else 0.0
    penalty = atype == "index" and sess.directional_permission <= 0.0
    if total == 0 or penalty:
        action, urgency = "AGUARDAR", "OBSERVAR"
    elif bull > bear:
        ratio = bull / total
        action, urgency = ("COMPRAR", "AGORA") if ratio > 0.70 and bull >= 4 else ("COMPRAR", "EM BREVE") if ratio > 0.60 else ("AGUARDAR", "OBSERVAR")
    elif bear > bull:
        ratio = bear / total
        action, urgency = ("VENDER", "AGORA") if ratio > 0.70 and bear >= 4 else ("VENDER", "EM BREVE") if ratio > 0.60 else ("AGUARDAR", "OBSERVAR")
    else:
        action, urgency = "AGUARDAR", "OBSERVAR"
    if reversal_risk == "ALTO" and action == "AGUARDAR" and health.status == "OK" and not index_market_closed_or_paused_for(symbol, sess):
        action, urgency = "SAIR · PROTEGER", "AGORA"
        msgs.insert(0, "Risco de reversão alto — proteger posição aberta antes de buscar entrada.")
    if index_market_closed_or_paused_for(symbol, sess):
        action, urgency = "AGUARDAR", "REVISÃO"
        msgs.insert(0, "Mercado/futuros fechados ou em pausa — leitura apenas retrospectiva; não emitir entrada nem saída operacional.")
    elif health.status in {"ATRASADO", "RUIM"}:
        msgs.insert(0, f"Saúde dos dados: {health.status}. {health.note}")
    if not msgs:
        msgs.append("Sem confluência clara — aguardar setup definido.")

    return CoachReport(symbol, atype, price, chg_pct, rsi_v, macd_h, sk_v, sd_v, atr_v, atr_pct, bb_pct, vrat_v,
        float(r["ema9"]), float(r["ema21"]), float(r["ema50"]), float(r["ema200"]), vwap_v,
        sup, res, action, urgency, trend, tstr, reversal_risk, sess, bull, bear, confidence, health, msgs)

# ─────────────────────────────────────────────────────────────────────────────
# ML ensemble temporal
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "rsi", "macd_hist", "atr_pct_feat", "bb_pos", "sk", "sd", "vrat", "vol_z", "atr_z",
    "ret_1", "ret_3", "ret_6", "ret_12", "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "ema9_dist", "ema21_dist", "ema50_dist", "ema100_dist", "ema200_dist", "vwap_dist",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Gera features sem exigir retorno futuro.

    Esta função é usada para previsão ao vivo. A versão anterior previa o
    último ponto que ainda tinha `future_ret` conhecido, o que podia deixar o
    sinal atrasado em `horizon_bars`. Agora treino/calibração e previsão live
    ficam separados: o treino usa `make_ml_frame`; a previsão atual usa este
    frame e pega a última linha com features disponíveis.
    """
    d = add_indicators(df).copy()
    bb_range = (d["bb_u"] - d["bb_l"]).replace(0, np.nan)
    d["bb_pos"] = (d["close"] - d["bb_l"]) / bb_range
    d["atr_pct_feat"] = d["atr"] / d["close"]
    for n in [9, 21, 50, 100, 200]:
        d[f"ema{n}_dist"] = d["close"] / d[f"ema{n}"] - 1
    d["vwap_dist"] = d["close"] / d["vwap"] - 1
    if isinstance(d.index, pd.DatetimeIndex):
        idx = d.index.tz_convert(ET) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(ET)
        hour_float = idx.hour + idx.minute / 60
        dow = idx.dayofweek
    else:
        hour_float = pd.Series(np.zeros(len(d)), index=d.index)
        dow = pd.Series(np.zeros(len(d)), index=d.index)
    d["hour_sin"] = np.sin(2 * np.pi * np.asarray(hour_float) / 24)
    d["hour_cos"] = np.cos(2 * np.pi * np.asarray(hour_float) / 24)
    d["dow_sin"] = np.sin(2 * np.pi * np.asarray(dow) / 7)
    d["dow_cos"] = np.cos(2 * np.pi * np.asarray(dow) / 7)
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def make_ml_frame(df: pd.DataFrame, horizon_bars: int = 3) -> pd.DataFrame:
    d = make_feature_frame(df)
    d["future_ret"] = d["close"].shift(-horizon_bars) / d["close"] - 1
    threshold = (d["atr"] / d["close"] * 0.50).clip(lower=0.0012, upper=0.012)
    d["target"] = np.where(d["future_ret"] > threshold, "COMPRAR", np.where(d["future_ret"] < -threshold, "VENDER", "AGUARDAR"))
    return d.dropna(subset=FEATURES + ["target", "future_ret"])

def _quality_shrink(prob: float, quality: float) -> float:
    if not np.isfinite(prob) or not np.isfinite(quality):
        return np.nan
    q = min(max((quality - 0.34) / 0.34, 0.0), 1.0)
    return 0.50 + (prob - 0.50) * q

# ─────────────────────────────────────────────────────────────────────────────
# Conformal Prediction (Adaptive Prediction Sets) — incerteza com garantia
# estatística, em vez de só um proxy heurístico (_quality_shrink).
#
# Diferença prática: calibrated_probability é um número único, otimista por
# natureza (mesmo encolhido). O conformal set responde "quais classes preciso
# considerar para ter X% de cobertura REAL, medida empiricamente contra dados
# que o modelo não usou para se calibrar". Quando o conjunto inclui mais de
# uma classe, isso é o sistema admitindo, de forma matematicamente honesta,
# que não há informação suficiente para descartar a alternativa.
#
# Implementação: split conformal com score APS (Adaptive Prediction Sets,
# Romano/Sesia/Candès 2020). Não depende de bibliotecas externas — só numpy.
# ─────────────────────────────────────────────────────────────────────────────

MIN_CONFORMAL_CAL_SIZE = 40  # abaixo disso, a calibração é estatisticamente frágil demais para confiar


def _aps_scores(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Para cada amostra, soma a probabilidade das classes mais prováveis até
    (e incluindo) a classe verdadeira. Esse é o 'conformal score' de cada ponto."""
    scores = np.zeros(len(y_true))
    class_index = {c: i for i, c in enumerate(classes)}
    for i in range(len(y_true)):
        order = np.argsort(-proba[i])
        sorted_p = proba[i][order]
        true_label = y_true[i] if isinstance(y_true, np.ndarray) else y_true.iloc[i]
        true_pos_in_order = np.where(classes[order] == true_label)[0]
        if len(true_pos_in_order) == 0:
            scores[i] = 1.0  # classe verdadeira fora do que o modelo conhece — pior caso
            continue
        true_pos = true_pos_in_order[0]
        scores[i] = float(sorted_p[: true_pos + 1].sum())
    return scores


def _aps_set_for_point(proba_row: np.ndarray, classes: np.ndarray, qhat: float) -> list[str]:
    """Monta o prediction set para um único ponto (a previsão atual)."""
    order = np.argsort(-proba_row)
    sorted_p = proba_row[order]
    cum = np.cumsum(sorted_p)
    k = int(np.searchsorted(cum, qhat) + 1)
    k = min(max(k, 1), len(classes))
    return [str(c) for c in classes[order[:k]]]


def conformal_calibrate(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    x_last: pd.DataFrame,
    classes: list[str],
    alpha: float = 0.10,
) -> dict:
    """Calibra um conformal predictor a partir de um conjunto de holdout
    (idealmente o último fold temporal, nunca usado para ajustar pesos do
    modelo) e aplica ao ponto mais recente.

    Retorna um dicionário pronto para alimentar os campos conformal_* do
    MLReport. Falha de forma segura (conformal_available=False) se a amostra
    de calibração for pequena demais para ter sentido estatístico.
    """
    out = {
        "available": False, "alpha": alpha, "set": [], "coverage_empirical": np.nan,
        "set_size_avg": np.nan, "note": "",
    }
    if len(X_cal) < MIN_CONFORMAL_CAL_SIZE:
        out["note"] = f"Conjunto de calibração pequeno demais ({len(X_cal)} < {MIN_CONFORMAL_CAL_SIZE}) — conformal não aplicado."
        return out
    try:
        proba_cal = model.predict_proba(X_cal)
        model_classes = np.array(_classes_of(model) or classes)
        if len(model_classes) == 0:
            out["note"] = "Modelo sem classes_ — conformal não aplicado."
            return out

        cal_scores = _aps_scores(proba_cal, y_cal.to_numpy(), model_classes)
        n_cal = len(cal_scores)
        # Correção de amostra finita (Romano et al. 2020): garante cobertura
        # >= 1-alpha mesmo com n_cal moderado, não só assintoticamente.
        q_level = float(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        qhat = float(np.quantile(cal_scores, min(q_level, 1.0), method="higher"))

        proba_last = model.predict_proba(x_last)[0]
        pred_set = _aps_set_for_point(proba_last, model_classes, qhat)

        # Cobertura empírica: nos próprios dados de calibração, qual fração
        # teria sido coberta com esse qhat. Serve como diagnóstico exibido ao
        # usuário — não recalibra nada, só mostra "isso bate com o prometido?".
        pred_sets_cal = [_aps_set_for_point(p, model_classes, qhat) for p in proba_cal]
        covered = sum(
            1 for s, yt in zip(pred_sets_cal, y_cal.to_numpy()) if str(yt) in s
        )
        coverage_emp = covered / n_cal
        avg_size = float(np.mean([len(s) for s in pred_sets_cal]))

        note = (
            f"Cobertura alvo {1-alpha:.0%}, medida em calibração {coverage_emp:.0%} "
            f"(n={n_cal}). Conjunto médio histórico: {avg_size:.1f} de {len(model_classes)} classes."
        )
        out.update({
            "available": True, "set": pred_set, "coverage_empirical": float(coverage_emp),
            "set_size_avg": avg_size, "note": note,
        })
        return out
    except Exception as exc:
        out["note"] = f"Falha no conformal: {exc}"
        return out

def _model_candidates(model_kind: str) -> list[tuple[str, Any]]:
    if RandomForestClassifier is None:
        return []
    candidates: list[tuple[str, Any]] = []
    if model_kind in {"Auto Ensemble", "Random Forest"}:
        candidates.append(("Random Forest", Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=380, max_depth=7, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ])))
    if model_kind in {"Auto Ensemble", "Extra Trees"} and ExtraTreesClassifier is not None:
        candidates.append(("Extra Trees", Pipeline([
            ("scale", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=420, max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=43, n_jobs=-1)),
        ])))
    if model_kind in {"Auto Ensemble", "Gradient Boosting"} and HistGradientBoostingClassifier is not None:
        candidates.append(("Gradient Boosting", HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=17, learning_rate=0.045, random_state=44)))
    if model_kind in {"Auto Ensemble", "Logístico"} and LogisticRegression is not None:
        candidates.append(("Logístico", Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", C=0.7)),
        ])))
    if not candidates and RandomForestClassifier is not None:
        candidates.append(("Random Forest", Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])))
    return candidates

def _classes_of(model: Any) -> list[str]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "classes_"):
        return list(model.named_steps["clf"].classes_)
    return []

# Sem cache: evita problemas de pickle em dataclasses ao editar o app
def train_predict_ml(symbol: str, period: str, interval: str, horizon_bars: int, model_kind: str = "Auto Ensemble") -> MLReport:
    if RandomForestClassifier is None:
        return MLReport(symbol, False, explanation="scikit-learn indisponível.")
    df = fetch(symbol, period=period, interval=interval)
    if df.empty or len(df) < 240:
        return MLReport(symbol, False, explanation="Histórico insuficiente para ML seguro.")
    d = make_ml_frame(df, horizon_bars=horizon_bars)
    X = d[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    y = d.loc[X.index, "target"]
    if len(X) < 180 or y.nunique() < 2:
        return MLReport(symbol, False, explanation="Amostra útil insuficiente ou pouca variação de classes.")
    balance = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}
    candidates = _model_candidates(model_kind)
    if not candidates:
        return MLReport(symbol, False, explanation="Nenhum estimador disponível.")

    splitter = TimeSeriesSplit(n_splits=4)
    leaderboard = []
    fitted_candidates = []
    # Guarda, por candidato, o modelo treinado apenas até o início do último
    # fold de teste — esse modelo nunca viu esse trecho de dado, então o
    # trecho serve como conjunto de calibração honesto para conformal
    # prediction, sem precisar de um treino extra dedicado.
    last_fold_holdout: dict[str, tuple[Any, pd.DataFrame, pd.Series]] = {}
    for name, proto in candidates:
        baccs, f1s, mccs = [], [], []
        fold_splits = list(splitter.split(X))
        for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
            try:
                import copy
                model = copy.deepcopy(proto)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict(X.iloc[test_idx])
                baccs.append(float(balanced_accuracy_score(y.iloc[test_idx], pred)))
                f1s.append(float(f1_score(y.iloc[test_idx], pred, average="macro", zero_division=0)))
                if matthews_corrcoef is not None:
                    mccs.append(float(matthews_corrcoef(y.iloc[test_idx], pred)))
                if fold_i == len(fold_splits) - 1:
                    last_fold_holdout[name] = (model, X.iloc[test_idx], y.iloc[test_idx])
            except Exception:
                continue
        if baccs:
            row = {"modelo": name, "balanced_accuracy": float(np.nanmean(baccs)), "f1_macro": float(np.nanmean(f1s)), "mcc": float(np.nanmean(mccs)) if mccs else np.nan}
            leaderboard.append(row)
            try:
                import copy
                final_model = copy.deepcopy(proto)
                final_model.fit(X, y)
                fitted_candidates.append((name, final_model, row["balanced_accuracy"]))
            except Exception:
                pass
    if not fitted_candidates:
        return MLReport(symbol, False, explanation="Falha no treino dos modelos candidatos.")

    # Ensemble ponderado por qualidade validada: modelos ruins têm peso mínimo.
    proba_acc: dict[str, float] = {}
    weight_sum = 0.0
    live_features = make_feature_frame(df)
    if live_features.empty:
        x_last = X.iloc[[-1]]
        live_note = "previsão no último ponto rotulado disponível"
    else:
        x_last = live_features[FEATURES].iloc[[-1]]
        live_note = "previsão no candle mais recente com features disponíveis"
    for name, model, quality in fitted_candidates:
        try:
            pred = model.predict(x_last)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_last)[0]
                classes = _classes_of(model)
                q_weight = max(quality - 0.33, 0.02)
                for c, p in zip(classes, probs):
                    proba_acc[str(c)] = proba_acc.get(str(c), 0.0) + float(p) * q_weight
                weight_sum += q_weight
            else:
                q_weight = max(quality - 0.33, 0.02)
                proba_acc[str(pred)] = proba_acc.get(str(pred), 0.0) + q_weight
                weight_sum += q_weight
        except Exception:
            continue
    if not proba_acc or weight_sum <= 0:
        return MLReport(symbol, False, explanation="Falha ao combinar probabilidades do ensemble.")
    probs_norm = {k: v / weight_sum for k, v in proba_acc.items()}
    prediction = max(probs_norm.items(), key=lambda kv: kv[1])[0]
    proba = float(probs_norm[prediction])
    quality = float(np.nanmean([r["balanced_accuracy"] for r in leaderboard]))
    f1m = float(np.nanmean([r["f1_macro"] for r in leaderboard]))
    mccv = float(np.nanmean([r["mcc"] for r in leaderboard])) if any(np.isfinite(r["mcc"]) for r in leaderboard) else np.nan
    calibrated = _quality_shrink(proba, quality)

    top_features: list[dict[str, Any]] = []
    try:
        best_name, best_model, _ = max(fitted_candidates, key=lambda t: t[2])
        if permutation_importance is not None and len(X) > 80:
            sample = X.tail(min(220, len(X)))
            sample_y = y.loc[sample.index]
            imp = permutation_importance(best_model, sample, sample_y, n_repeats=4, random_state=12, scoring="balanced_accuracy")
            order = np.argsort(imp.importances_mean)[::-1][:8]
            top_features = [{"feature": FEATURES[i], "importância": float(imp.importances_mean[i])} for i in order]
    except Exception:
        top_features = []

    explanation = "Auto Ensemble temporal: combina modelos com peso proporcional à qualidade walk-forward, encolhe probabilidade quando a qualidade é fraca e agora prevê no candle live mais recente. Modo: " + live_note + "."

    # Conformal prediction: usa o candidato de melhor qualidade que tenha um
    # holdout do último fold disponível (modelo nunca viu esses dados).
    conformal = {"available": False, "alpha": 0.10, "set": [], "coverage_empirical": np.nan, "set_size_avg": np.nan, "note": "Sem candidato elegível para conformal."}
    eligible = [(name, q) for name, _, q in fitted_candidates if name in last_fold_holdout]
    if eligible:
        best_name = max(eligible, key=lambda t: t[1])[0]
        cal_model, X_cal, y_cal = last_fold_holdout[best_name]
        conformal = conformal_calibrate(cal_model, X_cal, y_cal, x_last, _classes_of(cal_model), alpha=0.10)

    report = MLReport(symbol, True, prediction, proba, calibrated, quality, f1m, mccv, horizon_bars, balance, leaderboard, top_features, explanation)
    report.conformal_available = bool(conformal.get("available", False))
    report.conformal_alpha = float(conformal.get("alpha", 0.10))
    report.conformal_set = list(conformal.get("set", []))
    report.conformal_coverage_empirical = float(conformal.get("coverage_empirical", np.nan))
    report.conformal_set_size_avg = float(conformal.get("set_size_avg", np.nan))
    report.conformal_note = str(conformal.get("note", ""))
    return report

# ─────────────────────────────────────────────────────────────────────────────
# Backtest operacional e gates
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def walk_forward_backtest(symbol: str, period: str, interval: str, horizon_bars: int, model_kind: str,
                          min_train: int = 180, step_bars: int = 3, cost_bps: float = 2.0,
                          stop_atr: float = 0.75, target_atr: float = 1.25, slippage_bps: float = 1.0) -> tuple[pd.DataFrame, dict]:
    if RandomForestClassifier is None:
        return pd.DataFrame(), {"erro": "scikit-learn indisponível."}
    df = fetch(symbol, period=period, interval=interval)
    if df.empty:
        return pd.DataFrame(), {"erro": "Sem dados."}
    d = make_ml_frame(df, horizon_bars=horizon_bars)
    X = d[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    y = d.loc[X.index, "target"]
    if len(X) < min_train + step_bars + 20 or y.nunique() < 2:
        return pd.DataFrame(), {"erro": "Histórico insuficiente ou classes insuficientes."}
    candidates = _model_candidates(model_kind)
    if not candidates:
        return pd.DataFrame(), {"erro": "Nenhum modelo disponível."}

    rows = []
    step_bars = int(max(step_bars, horizon_bars, 1))
    for start in range(min_train, len(X) - step_bars + 1, step_bars):
        end = min(start + step_bars, len(X))
        # escolhe candidato com melhor validação interna simples no treino até o ponto atual
        best_model, best_name = None, ""
        best_score = -np.inf
        for name, proto in candidates:
            try:
                import copy
                inner = TimeSeriesSplit(n_splits=3)
                scores = []
                for tr, te in inner.split(X.iloc[:start]):
                    if y.iloc[:start].iloc[tr].nunique() < 2:
                        continue
                    m = copy.deepcopy(proto)
                    m.fit(X.iloc[:start].iloc[tr], y.iloc[:start].iloc[tr])
                    p = m.predict(X.iloc[:start].iloc[te])
                    scores.append(float(balanced_accuracy_score(y.iloc[:start].iloc[te], p)))
                score = float(np.nanmean(scores)) if scores else -np.inf
                if score > best_score:
                    best_score, best_model, best_name = score, copy.deepcopy(proto), name
            except Exception:
                continue
        if best_model is None:
            continue
        try:
            best_model.fit(X.iloc[:start], y.iloc[:start])
            pred = best_model.predict(X.iloc[start:end])
            if hasattr(best_model, "predict_proba"):
                conf = best_model.predict_proba(X.iloc[start:end]).max(axis=1)
            else:
                conf = np.repeat(np.nan, end - start)
        except Exception as exc:
            return pd.DataFrame(), {"erro": f"Falha no backtest: {exc}"}

        for i, idx in enumerate(X.iloc[start:end].index):
            decision = str(pred[i])
            trade = decision in {"COMPRAR", "VENDER"}
            entry = float(d.loc[idx, "close"])
            atr_now = float(d.loc[idx, "atr"])
            future_slice = d.loc[idx:].iloc[1:horizon_bars + 1]
            outcome = "SEM_TRADE"
            ret_pct = 0.0
            exit_price = entry
            if trade and not future_slice.empty and np.isfinite(atr_now):
                if decision == "COMPRAR":
                    stop = entry - stop_atr * atr_now
                    target = entry + target_atr * atr_now
                    for _, bar in future_slice.iterrows():
                        if float(bar["low"]) <= stop:
                            exit_price, outcome = stop, "STOP"; break
                        if float(bar["high"]) >= target:
                            exit_price, outcome = target, "ALVO"; break
                    if outcome == "SEM_TRADE":
                        exit_price, outcome = float(future_slice["close"].iloc[-1]), "TEMPO"
                    ret_pct = (exit_price / entry - 1) * 100
                else:
                    stop = entry + stop_atr * atr_now
                    target = entry - target_atr * atr_now
                    for _, bar in future_slice.iterrows():
                        if float(bar["high"]) >= stop:
                            exit_price, outcome = stop, "STOP"; break
                        if float(bar["low"]) <= target:
                            exit_price, outcome = target, "ALVO"; break
                    if outcome == "SEM_TRADE":
                        exit_price, outcome = float(future_slice["close"].iloc[-1]), "TEMPO"
                    ret_pct = (entry / exit_price - 1) * 100
                ret_pct -= (cost_bps + slippage_bps) / 100.0

            ts = idx
            ts_et = ts.tz_convert(ET).to_pydatetime() if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None else (ts.tz_localize("UTC").tz_convert(ET).to_pydatetime() if isinstance(ts, pd.Timestamp) else datetime.now(ET))
            sess_i = market_session(ts_et).name
            row_d = d.loc[idx]
            if float(row_d.get("ema9", np.nan)) > float(row_d.get("ema21", np.nan)) > float(row_d.get("ema50", np.nan)):
                tech_regime = "ALTA"
            elif float(row_d.get("ema9", np.nan)) < float(row_d.get("ema21", np.nan)) < float(row_d.get("ema50", np.nan)):
                tech_regime = "BAIXA"
            else:
                tech_regime = "LATERAL"
            atr_pct_ctx = float(row_d.get("atr_pct_feat", np.nan) * 100)
            vol_regime = "ALTA_VOL" if np.isfinite(atr_pct_ctx) and atr_pct_ctx >= 1.0 else "BAIXA/MÉDIA_VOL"
            rows.append({
                "time": idx, "symbol": symbol, "model": best_name, "prediction": decision,
                "true_class": str(y.loc[idx]), "confidence": float(conf[i]) if np.isfinite(conf[i]) else np.nan,
                "entry": entry, "exit_price": float(exit_price), "strategy_ret_pct": float(ret_pct),
                "trade": bool(trade), "hit": bool(ret_pct > 0) if trade else np.nan, "outcome": outcome,
                "session": sess_i, "hour_et": ts_et.hour + ts_et.minute / 60, "weekday": ts_et.strftime("%a"),
                "technical_regime": tech_regime, "vol_regime": vol_regime, "atr_pct": atr_pct_ctx,
            })
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {"erro": "Backtest não gerou previsões."}
    if bt["confidence"].notna().sum() >= 4:
        try:
            bt["confidence_band"] = pd.qcut(bt["confidence"], q=4, duplicates="drop").astype(str)
        except Exception:
            bt["confidence_band"] = "sem faixa"
    else:
        bt["confidence_band"] = "sem faixa"
    trade_bt = bt[bt["trade"]].copy()
    equity = bt["strategy_ret_pct"].cumsum()
    dd = equity - equity.cummax()
    gains = trade_bt.loc[trade_bt["strategy_ret_pct"] > 0, "strategy_ret_pct"].sum()
    losses = abs(trade_bt.loc[trade_bt["strategy_ret_pct"] < 0, "strategy_ret_pct"].sum())
    metrics = {
        "ativo": symbol, "modelo": model_kind, "observações": int(len(bt)), "operações": int(len(trade_bt)),
        "taxa_operação": float(len(trade_bt) / len(bt)) if len(bt) else np.nan,
        "hit_rate": float(trade_bt["hit"].mean()) if len(trade_bt) else np.nan,
        "retorno_médio_%": float(trade_bt["strategy_ret_pct"].mean()) if len(trade_bt) else np.nan,
        "retorno_total_%": float(bt["strategy_ret_pct"].sum()), "drawdown_%": float(dd.min()) if len(dd) else np.nan,
        "profit_factor": float(gains / losses) if losses > 0 else np.inf if gains > 0 else np.nan,
        "stop_atr": float(stop_atr), "target_atr": float(target_atr), "custo+slippage_bps": float(cost_bps + slippage_bps),
    }
    return bt, metrics

def _context_summary(bt: pd.DataFrame, by: str) -> pd.DataFrame:
    if bt.empty or by not in bt.columns:
        return pd.DataFrame()
    trades = bt[bt["trade"]].copy()
    if trades.empty:
        return pd.DataFrame()
    out = trades.groupby(by, dropna=False).agg(
        operações=("trade", "size"), hit_rate=("hit", "mean"), retorno_médio_pct=("strategy_ret_pct", "mean"),
        retorno_total_pct=("strategy_ret_pct", "sum"), confiança_média=("confidence", "mean"), atr_médio_pct=("atr_pct", "mean"),
    ).reset_index()
    total_obs = bt.groupby(by, dropna=False).size().rename("observações").reset_index()
    out = out.merge(total_obs, on=by, how="left")
    out["taxa_operação"] = out["operações"] / out["observações"]
    return out.sort_values(["retorno_total_pct", "hit_rate"], ascending=False)

@st.cache_data(ttl=900, show_spinner=False)
def learned_context_gates(symbols: tuple[str, ...], period: str, interval: str, horizon_bars: int, model_kind: str,
                          min_train: int, step_bars: int, cost_bps: float, stop_atr: float, target_atr: float,
                          slippage_bps: float, strictness: float) -> dict:
    poor: dict[str, list[str]] = {}
    good: dict[str, list[str]] = {}
    metrics = []
    for sym in symbols:
        bt, mt = walk_forward_backtest(sym, period, interval, horizon_bars, model_kind, min_train, step_bars, cost_bps, stop_atr, target_atr, slippage_bps)
        if bt.empty:
            continue
        metrics.append(mt)
        for ctx in ["session", "technical_regime", "prediction", "vol_regime"]:
            tab = _context_summary(bt, ctx)
            if tab.empty:
                continue
            for _, row in tab.iterrows():
                label = str(row[ctx])
                ops = int(row.get("operações", 0))
                hit = float(row.get("hit_rate", np.nan))
                ret = float(row.get("retorno_médio_pct", np.nan))
                if ops < MIN_CONTEXT_OPS:
                    continue
                if np.isfinite(hit) and np.isfinite(ret) and (ret < POOR_CONTEXT_RET_PCT * (0.75 + strictness) or hit < POOR_CONTEXT_HIT - 0.02 * (1 - strictness)):
                    poor.setdefault(ctx, []).append(label)
                if np.isfinite(hit) and np.isfinite(ret) and (ret > GOOD_CONTEXT_RET_PCT * (1.25 - strictness * 0.25) and hit > GOOD_CONTEXT_HIT):
                    good.setdefault(ctx, []).append(label)
    poor = {k: sorted(set(v)) for k, v in poor.items()}
    good = {k: sorted(set(v)) for k, v in good.items()}
    return {"available": bool(metrics), "poor": poor, "good": good, "metrics": metrics, "reason": "OK" if metrics else "Sem backtest suficiente para gates."}

# ─────────────────────────────────────────────────────────────────────────────
# Agentes, síntese e risco
# ─────────────────────────────────────────────────────────────────────────────

def technical_agent(r: CoachReport) -> AgentOpinion:
    if r.score_bull > r.score_bear and r.confidence_rule >= 0.35:
        return AgentOpinion("Técnico", "COMPRAR", r.confidence_rule, f"Confluência compradora {r.score_bull}×{r.score_bear}.")
    if r.score_bear > r.score_bull and r.confidence_rule >= 0.35:
        return AgentOpinion("Técnico", "VENDER", r.confidence_rule, f"Confluência vendedora {r.score_bear}×{r.score_bull}.")
    return AgentOpinion("Técnico", "AGUARDAR", 0.45, "Sinais técnicos mistos.")

def risk_agent(r: CoachReport) -> AgentOpinion:
    if index_market_closed_or_paused_for(r.symbol, r.session):
        return AgentOpinion("Risco", "AGUARDAR", 0.95, "Mercado fechado/pausado: dados são retrospectivos; não há sinal operacional ao vivo.")
    if r.data_health.status in {"RUIM", "ATRASADO", "FECHADO"}:
        return AgentOpinion("Risco", "AGUARDAR", 0.85, f"Dados {r.data_health.status.lower()}.")
    if r.reversal_risk == "ALTO":
        return AgentOpinion("Risco", "SAIR · PROTEGER", 0.92, "Risco de reversão alto.")
    if r.asset_type == "index" and r.atr_pct > 1.30:
        return AgentOpinion("Risco", "AGUARDAR", 0.72, f"ATR elevado ({r.atr_pct:.2f}%).")
    if r.reversal_risk == "MÉDIO":
        return AgentOpinion("Risco", "AGUARDAR", 0.55, "Risco intermediário; exigir confirmação.")
    return AgentOpinion("Risco", "AGUARDAR", 0.30, "Sem alerta extremo.")

def session_agent(r: CoachReport) -> AgentOpinion:
    if r.asset_type == "index" and r.session.directional_permission <= 0:
        return AgentOpinion("Sessão", "AGUARDAR", 0.90, r.session.note)
    if r.asset_type == "index" and r.session.directional_permission < 0.60:
        vote = r.action if r.action != "SAIR · PROTEGER" else "AGUARDAR"
        return AgentOpinion("Sessão", vote, 0.40, r.session.note)
    if r.session.name in {"MANHÃ", "TARDE"}:
        vote = r.action if r.action != "SAIR · PROTEGER" else "AGUARDAR"
        return AgentOpinion("Sessão", vote, 0.58, "Janela operacional favorável.")
    return AgentOpinion("Sessão", "AGUARDAR", 0.50, r.session.note)

def ml_agent(m: MLReport) -> AgentOpinion:
    if not m.available:
        return AgentOpinion("ML", "AGUARDAR", 0.25, m.explanation)
    conf = float(m.calibrated_probability if np.isfinite(m.calibrated_probability) else m.probability)
    if not np.isfinite(m.model_quality) or m.model_quality < 0.42:
        return AgentOpinion("ML", "AGUARDAR", 0.58, f"Qualidade temporal fraca ({m.model_quality:.2f}).")
    if conf < 0.51:
        return AgentOpinion("ML", "AGUARDAR", 0.50, f"Confiança insuficiente ({conf:.0%}).")
    return AgentOpinion("ML", m.prediction, min(max(conf, 0.0), 0.88), f"p ajustada {conf:.0%}; qualidade {m.model_quality:.2f}.")



def conformal_uncertainty_agent(m: MLReport) -> AgentOpinion:
    """Transforma incerteza conformal em voto operacional.

    Conjunto unitário reforça o ML. Conjunto amplo, conjunto com COMPRAR e
    VENDER simultaneamente ou conjunto que não contém a previsão pontual
    rebaixa para AGUARDAR.
    """
    if not m.available:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.28, "ML indisponível; sem conjunto conformal.")
    if not m.conformal_available:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.34, m.conformal_note or "Conformal indisponível.")
    s = set(str(x) for x in m.conformal_set)
    if not s:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.60, "Conjunto conformal vazio; bloquear confiança direcional.")
    if m.prediction not in s:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.76, f"Conjunto conformal {sorted(s)} não contém a previsão pontual {m.prediction}.")
    if "COMPRAR" in s and "VENDER" in s:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.84, f"Conformal não descarta compra e venda: {sorted(s)}.")
    if len(s) >= 3:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.82, f"Conjunto amplo demais: {sorted(s)}.")
    if len(s) == 2:
        return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.58, f"Incerteza moderada: {sorted(s)}.")
    only = next(iter(s))
    if only in {"COMPRAR", "VENDER"}:
        return AgentOpinion("Incerteza conformal", only, 0.42, f"Conjunto conformal unitário: {only}. {m.conformal_note}")
    return AgentOpinion("Incerteza conformal", "AGUARDAR", 0.50, f"Conjunto unitário é AGUARDAR. {m.conformal_note}")



def _beta_lower_approx(alpha: float, beta: float, z: float = 1.28) -> float:
    total = alpha + beta
    if total <= 0:
        return 0.0
    mean = alpha / total
    var = (alpha * beta) / ((total ** 2) * (total + 1))
    return max(0.0, min(1.0, mean - z * math.sqrt(max(var, 0.0))))


def _bayes_prior_params(r: CoachReport, ml: MLReport, strategy: TimeframeStrategy, prior_profile: str = "Conservador") -> tuple[float, float, str]:
    """Priors interpretáveis para o agente Bayes contextual.

    O prior é deliberadamente fraco: ele só orienta quando ainda não há diário
    suficiente. A evidência real por contexto domina assim que o operador fecha
    trades no app.
    """
    profile = str(prior_profile or "Conservador").lower()
    base = 0.50
    strength = 8.0
    notes: list[str] = []

    if r.asset_type == "forex":
        tier = forex_stability_label(r.symbol)
        if tier == "estável":
            base += 0.025; notes.append("prior favorece levemente FX estável")
        elif tier == "volátil":
            base -= 0.030; notes.append("prior penaliza FX volátil")
    elif r.asset_type == "index":
        if r.session.name in {"MANHÃ", "TARDE"}:
            base += 0.015; notes.append("sessão favorável para índice")
        elif r.session.directional_permission < 0.60:
            base -= 0.030; notes.append("sessão menos permissiva")

    if strategy.minutes <= 5:
        base -= 0.020; strength += 2.0; notes.append("scalp curto exige prudência")
    elif strategy.minutes >= 120:
        base -= 0.010; strength += 1.0; notes.append("janela longa exige tese robusta")

    if ml.available and np.isfinite(ml.model_quality):
        base += max(min((float(ml.model_quality) - 0.50) * 0.16, 0.045), -0.045)
        notes.append(f"qualidade ML={ml.model_quality:.2f}")

    if profile.startswith("neut"):
        strength = max(5.0, strength - 3.0)
        notes.append("prior neutro/mais maleável")
    elif profile.startswith("agress"):
        base += 0.015
        strength = max(4.0, strength - 4.0)
        notes.append("prior agressivo: evidência pesa mais rápido")
    else:
        base -= 0.005
        strength += 3.0
        notes.append("prior conservador")

    base = float(min(max(base, 0.42), 0.58))
    strength = float(min(max(strength, 4.0), 18.0))
    return base, strength, "; ".join(notes)


def _closed_trades_for_bayes() -> pd.DataFrame:
    """Carrega trades fechados e normaliza colunas usadas pelo Bayes contextual."""
    tlog = load_trade_log()
    required = {"symbol", "side", "status", "realized_pnl"}
    if tlog.empty or not required.issubset(set(tlog.columns)):
        return pd.DataFrame()
    closed = tlog[tlog["status"].astype(str).str.upper() == "FECHADA"].copy()
    if closed.empty:
        return closed
    closed["symbol"] = closed["symbol"].astype(str).str.upper()
    closed["side"] = closed["side"].astype(str).str.upper()
    closed["realized_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce")
    closed = closed.dropna(subset=["realized_pnl"])
    if "timeframe_minutes" in closed.columns:
        closed["timeframe_minutes"] = pd.to_numeric(closed["timeframe_minutes"], errors="coerce")
    else:
        closed["timeframe_minutes"] = np.nan
    if "session" not in closed.columns:
        closed["session"] = ""
    if "trend" not in closed.columns:
        closed["trend"] = ""
    return closed


def _bayes_context_rows(closed: pd.DataFrame, r: CoachReport, side: str, strategy: TimeframeStrategy) -> list[tuple[str, pd.DataFrame, float]]:
    """Retorna evidências hierárquicas do contexto mais específico ao mais geral.

    Camadas usadas:
    1. ativo × janela × sessão × direção;
    2. ativo × janela × direção;
    3. ativo × sessão × direção;
    4. ativo × direção;
    5. família/tier × janela × direção;
    6. direção global.

    A ponderação evita overfitting: contexto muito específico conta mais, mas
    contextos mais amplos entram como informação parcial quando a amostra é
    pequena.
    """
    if closed.empty:
        return []
    sym = str(r.symbol).upper()
    side = str(side).upper()
    tf = int(strategy.minutes)
    sess = str(r.session.name)
    out: list[tuple[str, pd.DataFrame, float]] = []

    side_df = closed[closed["side"] == side].copy()
    if side_df.empty:
        return []

    exact = side_df[(side_df["symbol"] == sym) & (side_df["timeframe_minutes"] == tf) & (side_df["session"].astype(str) == sess)]
    out.append(("ativo×janela×sessão×direção", exact, 1.00))

    by_tf = side_df[(side_df["symbol"] == sym) & (side_df["timeframe_minutes"] == tf)]
    out.append(("ativo×janela×direção", by_tf, 0.70))

    by_sess = side_df[(side_df["symbol"] == sym) & (side_df["session"].astype(str) == sess)]
    out.append(("ativo×sessão×direção", by_sess, 0.55))

    by_symbol = side_df[side_df["symbol"] == sym]
    out.append(("ativo×direção", by_symbol, 0.40))

    if r.asset_type == "forex":
        same_family_symbols = [k for k, v in FOREX_STABILITY_TIER.items() if v == forex_stability_label(r.symbol)]
    elif r.asset_type == "index":
        same_family_symbols = DEFAULT_INDEX_SYMBOLS
    else:
        same_family_symbols = [sym]
    family = side_df[(side_df["symbol"].isin([x.upper() for x in same_family_symbols])) & (side_df["timeframe_minutes"] == tf)]
    out.append(("família/tier×janela×direção", family, 0.28))

    out.append(("direção global", side_df, 0.18))
    return out


def _weighted_beta_from_context(closed: pd.DataFrame, r: CoachReport, side: str, strategy: TimeframeStrategy, alpha: float, beta: float) -> tuple[float, float, float, float, list[str]]:
    """Atualiza Beta com evidência ponderada e retorna também EV/payoff."""
    n_eff = 0.0
    weighted_pnl_sum = 0.0
    weighted_pnl_n = 0.0
    notes: list[str] = []
    seen_ids: set[str] = set()
    for label, frame, weight in _bayes_context_rows(closed, r, side, strategy):
        if frame.empty:
            continue
        # Evita que a mesma operação conte integralmente em várias camadas:
        # contextos amplos ainda ajudam, mas com peso reduzido e sem duplicar
        # demais as evidências mais específicas.
        if "id" in frame.columns:
            frame = frame[~frame["id"].astype(str).isin(seen_ids)].copy()
            seen_ids.update(set(frame.get("id", pd.Series(dtype=str)).astype(str)))
        if frame.empty:
            continue
        pnl = pd.to_numeric(frame["realized_pnl"], errors="coerce").dropna()
        if pnl.empty:
            continue
        n = float(len(pnl)) * float(weight)
        wins = float((pnl > 0).sum()) * float(weight)
        alpha += wins
        beta += n - wins
        n_eff += n
        weighted_pnl_sum += float(pnl.mean()) * n
        weighted_pnl_n += n
        notes.append(f"{label}: n={len(pnl)}, win={(pnl > 0).mean():.0%}, peso={weight:.2f}")
    ev_cash = weighted_pnl_sum / weighted_pnl_n if weighted_pnl_n > 0 else np.nan
    return alpha, beta, n_eff, ev_cash, notes[:5]


def bayesian_context_agent(r: CoachReport, ml: MLReport, strategy: TimeframeStrategy, prior_profile: str = "Conservador") -> AgentOpinion:
    """Agente bayesiano 9.9: Beta-Binomial hierárquico por contexto real.

    Usa evidência fechada do diário por ativo × janela × sessão × direção,
    recuando gradualmente para contextos mais amplos quando a amostra é baixa.
    A saída considera posterior média, limite conservador e EV financeiro médio
    observado no contexto. Não opera sozinho: serve como freio/reforço no consenso.
    """
    candidate = r.action if r.action in {"COMPRAR", "VENDER"} else (ml.prediction if ml.available and ml.prediction in {"COMPRAR", "VENDER"} else "AGUARDAR")
    if candidate not in {"COMPRAR", "VENDER"}:
        return AgentOpinion("Bayes contexto 9.9", "AGUARDAR", 0.38, "Sem direção candidata para atualizar posterior contextual.")

    prior_mean, prior_strength, prior_note = _bayes_prior_params(r, ml, strategy, prior_profile)
    alpha = max(0.5, prior_mean * prior_strength)
    beta = max(0.5, (1.0 - prior_mean) * prior_strength)
    closed = _closed_trades_for_bayes()
    alpha, beta, n_eff, ev_cash, evidence_notes = _weighted_beta_from_context(closed, r, candidate, strategy, alpha, beta)

    post_mean = alpha / (alpha + beta)
    post_low80 = _beta_lower_approx(alpha, beta, z=1.28)
    post_low90 = _beta_lower_approx(alpha, beta, z=1.64)
    ev_txt = "EV s/ histórico" if not np.isfinite(ev_cash) else f"EV≈{ev_cash:+.2f}"
    evidence_txt = "; ".join(evidence_notes) if evidence_notes else "sem trades reais fechados neste contexto"
    detail = (
        f"prior={prior_mean:.0%}/força={prior_strength:.1f} ({prior_note}); "
        f"posterior={post_mean:.0%}, low80≈{post_low80:.0%}, low90≈{post_low90:.0%}, "
        f"n efetivo={n_eff:.1f}, {ev_txt}. Evidência: {evidence_txt}."
    )

    # Regras prudenciais 9.9: com evidência real razoável, Bayes vira gate; com
    # pouca evidência, atua como prior fraco e não cria excesso de confiança.
    if n_eff >= 10 and (post_low80 < 0.48 or (np.isfinite(ev_cash) and ev_cash <= 0)):
        return AgentOpinion("Bayes contexto 9.9", "AGUARDAR", 0.86, f"Contexto real desfavorável para {r.symbol} {candidate}; {detail}")
    if n_eff >= 6 and post_low80 >= 0.53 and (not np.isfinite(ev_cash) or ev_cash > 0) and r.reversal_risk != "ALTO":
        return AgentOpinion("Bayes contexto 9.9", candidate, 0.58, f"Contexto real favorece {candidate}; {detail}")
    if n_eff >= 3 and post_mean >= 0.56 and (not np.isfinite(ev_cash) or ev_cash >= 0) and r.reversal_risk == "BAIXO":
        return AgentOpinion("Bayes contexto 9.9", candidate, 0.42, f"Sinal bayesiano inicial favorável; {detail}")
    if post_mean < 0.50 or r.reversal_risk == "ALTO":
        return AgentOpinion("Bayes contexto 9.9", "AGUARDAR", 0.62, f"Posterior não sustenta entrada; {detail}")
    return AgentOpinion("Bayes contexto 9.9", "AGUARDAR", 0.44, f"Evidência contextual ainda insuficiente; {detail}")

def relative_strength_agent(reports: list[CoachReport], current: CoachReport) -> AgentOpinion:
    idx = [r for r in reports if r.asset_type == "index"]
    if current.asset_type != "index" or len(idx) < 2:
        return AgentOpinion("Força relativa", "AGUARDAR", 0.25, "Não aplicável.")
    ordered = sorted(idx, key=lambda r: r.change_pct, reverse=True)
    if current.symbol == ordered[0].symbol and current.change_pct > 0:
        return AgentOpinion("Força relativa", "COMPRAR", 0.50, f"{current.symbol} lidera positivamente.")
    if current.symbol == ordered[-1].symbol and current.change_pct < 0:
        return AgentOpinion("Força relativa", "VENDER", 0.50, f"{current.symbol} é o mais fraco.")
    return AgentOpinion("Força relativa", "AGUARDAR", 0.35, "Sem liderança clara.")

def forex_stability_label(symbol: str) -> str:
    return FOREX_STABILITY_TIER.get(symbol.upper(), "não classificado")

def forex_stability_note(symbol: str) -> str:
    tier = forex_stability_label(symbol)
    profile = FOREX_TIER_PROFILE.get(tier, {})
    if not profile:
        return "Sem classificação específica de estabilidade para este par."
    return f"{tier.title()}: {profile.get('note', '')} Estratégia: {profile.get('bias', '')}."

def forex_stability_agent(r: CoachReport) -> AgentOpinion:
    """Agente leve para diferenciar pares estáveis, intermediários e voláteis.

    Não tenta prever direção; apenas ajusta a prudência. Pares estáveis podem
    aceitar confluência moderada quando risco e dados estão bons. Pares mais
    voláteis exigem confirmação adicional.
    """
    if r.asset_type != "forex":
        return AgentOpinion("Estabilidade FX", "AGUARDAR", 0.18, "Não aplicável a índices/commodities.")
    tier = forex_stability_label(r.symbol)
    if tier == "estável":
        if r.data_health.status == "OK" and r.reversal_risk == "BAIXO" and r.action in {"COMPRAR", "VENDER"}:
            return AgentOpinion("Estabilidade FX", r.action, 0.38, forex_stability_note(r.symbol))
        return AgentOpinion("Estabilidade FX", "AGUARDAR", 0.36, f"{forex_stability_note(r.symbol)} Exigir dados OK e risco baixo.")
    if tier == "intermediário":
        if r.action in {"COMPRAR", "VENDER"} and r.trend_strength == "FORTE" and r.reversal_risk == "BAIXO":
            return AgentOpinion("Estabilidade FX", r.action, 0.30, forex_stability_note(r.symbol))
        return AgentOpinion("Estabilidade FX", "AGUARDAR", 0.40, f"{forex_stability_note(r.symbol)} Aguardar confirmação adicional.")
    if tier == "volátil":
        return AgentOpinion("Estabilidade FX", "AGUARDAR", 0.48, forex_stability_note(r.symbol))
    return AgentOpinion("Estabilidade FX", "AGUARDAR", 0.28, "Par FX sem tier de estabilidade definido.")

def macro_agent(ctx: ContextReport, r: CoachReport) -> AgentOpinion:
    if r.asset_type != "index":
        return AgentOpinion("Macro", "AGUARDAR", 0.30, "Macro usado principalmente para índices.")
    score = ctx.risk_score + (-ctx.news_score)
    if ctx.risk_regime == "DEFENSIVO" or ctx.news_pressure == "RISK-OFF" or score >= 1.20:
        return AgentOpinion("Macro", "AGUARDAR", 0.78, f"Regime {ctx.risk_regime}; notícias {ctx.news_pressure}.")
    if ctx.risk_regime == "CONSTRUTIVO" and ctx.news_pressure != "RISK-OFF":
        return AgentOpinion("Macro", "COMPRAR", 0.45, "Contexto construtivo.")
    return AgentOpinion("Macro", "AGUARDAR", 0.45, f"Regime {ctx.risk_regime}; notícias {ctx.news_pressure}.")


def timeframe_agent(strategy: TimeframeStrategy, r: CoachReport) -> AgentOpinion:
    """Agente que ajusta prudência conforme a janela operacional escolhida."""
    if r.asset_type == "index" and r.session.directional_permission < strategy.min_session_permission:
        return AgentOpinion(
            "Janela temporal",
            "AGUARDAR",
            0.62 + min(0.20, strategy.min_session_permission - r.session.directional_permission),
            f"{strategy.label}: sessão {r.session.name} não oferece liquidez/permissão suficiente para este horizonte.",
        )
    if strategy.minutes <= 5:
        if r.reversal_risk == "ALTO" or r.data_health.status != "OK":
            return AgentOpinion("Janela temporal", "AGUARDAR", 0.70, f"{strategy.label}: scalp exige dados OK e baixo risco de reversão.")
        vote = r.action if r.action in {"COMPRAR", "VENDER"} else "AGUARDAR"
        return AgentOpinion("Janela temporal", vote, 0.44, f"{strategy.label}: operar apenas com execução rápida e stop rígido.")
    if strategy.minutes in {15, 30}:
        if r.trend == "LATERAL" and r.reversal_risk != "BAIXO":
            return AgentOpinion("Janela temporal", "AGUARDAR", 0.58, f"{strategy.label}: lateralidade sem reversão clara enfraquece a tese.")
        vote = r.action if r.action in {"COMPRAR", "VENDER"} else "AGUARDAR"
        return AgentOpinion("Janela temporal", vote, 0.48, f"{strategy.label}: favorece momentum/reteste, não impulso isolado.")
    # 60, 120, 240: exigir tese mais estrutural.
    if r.trend_strength == "FRACA" or r.reversal_risk == "ALTO":
        return AgentOpinion("Janela temporal", "AGUARDAR", 0.64, f"{strategy.label}: janelas longas exigem regime mais claro.")
    vote = r.action if r.action in {"COMPRAR", "VENDER"} else "AGUARDAR"
    return AgentOpinion("Janela temporal", vote, 0.52, f"{strategy.label}: tese deve sobreviver a ruído de vários candles.")

def synthesize_agents(opinions: list[AgentOpinion], r: CoachReport, ml: MLReport, ctx: ContextReport, gates: dict, strictness: float) -> tuple[str, float, str, dict[str, float], DecisionAudit]:
    weights = {"COMPRAR": 0.0, "VENDER": 0.0, "AGUARDAR": 0.0, "SAIR · PROTEGER": 0.0}
    components = {}
    for op in opinions:
        weights[op.vote] = weights.get(op.vote, 0.0) + float(op.confidence)
        components[op.name] = float(op.confidence) * (1 if op.vote == "COMPRAR" else -1 if op.vote == "VENDER" else 0)
    market_blocked = index_market_closed_or_paused_for(r.symbol, r.session)
    if market_blocked:
        raw_decision, raw_score, rationale = "AGUARDAR", max(weights.get("AGUARDAR", 0.0), 0.95), "Mercado fechado/pausado: leitura apenas retrospectiva; coaching operacional bloqueado."
    elif weights.get("SAIR · PROTEGER", 0.0) >= 0.78:
        raw_decision, raw_score, rationale = "SAIR · PROTEGER", weights["SAIR · PROTEGER"], "Risco domina."
    else:
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        raw_decision, raw_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = raw_score - second
        min_margin = 0.30 + 0.25 * strictness
        min_score = 1.00 + 0.25 * strictness
        if raw_decision in {"COMPRAR", "VENDER"} and (margin < min_margin or raw_score < min_score):
            raw_decision, raw_score, rationale = "AGUARDAR", max(raw_score, 0.50), "Margem/score insuficientes."
        else:
            rationale = "Consenso ponderado entre técnica, risco, sessão, ML, conformal, Bayes contextual, força relativa e macro."
    final_decision, final_score = raw_decision, float(raw_score)
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else float(ranked[0][1])
    blockers, boosts = [], []

    # Gates duros de integridade
    if r.data_health.status in {"RUIM", "ATRASADO"} and final_decision in {"COMPRAR", "VENDER"}:
        blockers.append(f"dados {r.data_health.status.lower()}"); final_decision = "AGUARDAR"; final_score = max(0.60, final_score * 0.65)
    if market_blocked and final_decision != "AGUARDAR":
        blockers.append("futuros fechados/pausados: sinal ao vivo bloqueado"); final_decision = "AGUARDAR"; final_score = max(0.60, final_score * 0.55)
    if final_decision in {"COMPRAR", "VENDER"} and ml.available and ml.prediction in {"COMPRAR", "VENDER"} and ml.prediction != final_decision and np.isfinite(ml.calibrated_probability) and ml.calibrated_probability >= 0.58:
        blockers.append(f"ML discorda com p={ml.calibrated_probability:.0%}"); final_decision = "AGUARDAR"; final_score = max(0.65, final_score * 0.75)
    if final_decision == "COMPRAR" and ctx.risk_regime == "DEFENSIVO" and r.asset_type == "index":
        blockers.append("macro defensivo contra compra agressiva"); final_decision = "AGUARDAR"; final_score = max(0.65, final_score * 0.75)

    # Gate conformal duro: sinal direcional só passa quando a incerteza validada
    # não contém a direção oposta nem um conjunto amplo demais.
    if final_decision in {"COMPRAR", "VENDER"} and ml.available and ml.conformal_available:
        cset = set(str(x) for x in ml.conformal_set)
        opposite = "VENDER" if final_decision == "COMPRAR" else "COMPRAR"
        if not cset:
            blockers.append("conformal sem conjunto válido")
            final_decision = "AGUARDAR"; final_score = max(0.65, final_score * 0.65)
        elif final_decision not in cset:
            blockers.append(f"conformal não contém {final_decision}: {sorted(cset)}")
            final_decision = "AGUARDAR"; final_score = max(0.65, final_score * 0.62)
        elif opposite in cset or len(cset) >= 3:
            blockers.append(f"conformal ainda admite cenário oposto/amplo: {sorted(cset)}")
            final_decision = "AGUARDAR"; final_score = max(0.65, final_score * 0.68)
        elif len(cset) == 2 and "AGUARDAR" in cset:
            blockers.append(f"conformal ainda admite AGUARDAR: {sorted(cset)}")
            final_score = max(0.70, final_score * (0.82 - 0.12 * strictness))
        elif len(cset) == 1:
            boosts.append(f"conformal unitário: {next(iter(cset))}")
            final_score = min(final_score + 0.14, final_score * 1.06)

    # Gate Bayes 9.9: quando o histórico real contextual é suficientemente
    # desfavorável, ele bloqueia mesmo que o consenso bruto tenha passado.
    if final_decision in {"COMPRAR", "VENDER"}:
        for op in opinions:
            if str(op.name).startswith("Bayes contexto") and op.vote == "AGUARDAR" and op.confidence >= 0.80:
                blockers.append("Bayes contextual desfavorável: " + op.reason[:180])
                final_decision = "AGUARDAR"
                final_score = max(0.65, final_score * (0.60 - 0.10 * strictness))
                break

    # Gates contextuais aprendidos
    if gates.get("available") and final_decision in {"COMPRAR", "VENDER"}:
        poor, good = gates.get("poor", {}), gates.get("good", {})
        current_context = {
            "session": r.session.name,
            "technical_regime": r.trend,
            "prediction": final_decision,
            "vol_regime": "ALTA_VOL" if r.atr_pct >= 1.0 else "BAIXA/MÉDIA_VOL",
        }
        for ctx_name, label in current_context.items():
            if label in set(poor.get(ctx_name, [])):
                blockers.append(f"contexto ruim histórico: {ctx_name}={label}")
                final_decision = "AGUARDAR"
                final_score = max(0.65, final_score * (0.65 - 0.20 * strictness))
            elif label in set(good.get(ctx_name, [])):
                boosts.append(f"contexto bom histórico: {ctx_name}={label}")
                final_score = min(final_score * (1.04 + 0.08 * (1 - strictness)), final_score + 0.18)

    reason = rationale
    if blockers:
        reason = "Sinal rebaixado por: " + "; ".join(blockers)
    elif boosts and final_decision in {"COMPRAR", "VENDER"}:
        reason = rationale + " Reforço: " + "; ".join(boosts)
    audit = DecisionAudit(raw_decision, float(raw_score), final_decision, float(final_score), margin, blockers, boosts, components, reason)
    return final_decision, float(final_score), reason, weights, audit

def build_risk_plan(r: CoachReport, decision: str, risk_cash: float, stop_atr: float, target_atr: float) -> RiskPlan:
    entry = r.price
    if decision == "COMPRAR":
        stop = entry - stop_atr * r.atr
        target = entry + target_atr * r.atr
        invalid = "Perde o stop ou fecha abaixo do VWAP/EMA21 com aumento de volatilidade."
    elif decision == "VENDER":
        stop = entry + stop_atr * r.atr
        target = entry - target_atr * r.atr
        invalid = "Rompe o stop ou fecha acima do VWAP/EMA21 com recuperação de força."
    else:
        stop, target = np.nan, np.nan
        invalid = "Sem entrada; observar nova confluência."
    risk_points = abs(entry - stop) if np.isfinite(stop) else np.nan
    reward_points = abs(target - entry) if np.isfinite(target) else np.nan
    rr = reward_points / risk_points if np.isfinite(risk_points) and risk_points > 0 else np.nan
    units = risk_cash / risk_points if np.isfinite(risk_points) and risk_points > 0 else 0.0
    return RiskPlan(r.symbol, decision, entry, stop, target, risk_points, reward_points, rr, units, risk_cash, invalid)

# ─────────────────────────────────────────────────────────────────────────────
# Diário e alertas
# ─────────────────────────────────────────────────────────────────────────────

def fmt_price(x: float, atype: str = "index") -> str:
    if not np.isfinite(x): return "—"
    if atype in {"forex"}: return f"{x:,.5f}"
    if atype in {"index", "rates"}: return f"{x:,.2f}"
    return f"{x:,.2f}"

def action_badge(action: str) -> str:
    return {"COMPRAR": "🟢 COMPRAR", "VENDER": "🔴 VENDER", "AGUARDAR": "🟡 AGUARDAR", "SAIR · PROTEGER": "🛡️ SAIR · PROTEGER"}.get(action, action)

def signal_id(symbol: str, ts: str, decision: str, price: float) -> str:
    return hashlib.sha1(f"{symbol}|{ts}|{decision}|{price:.6f}".encode()).hexdigest()[:12]

def load_journal() -> pd.DataFrame:
    if JOURNAL_PATH.exists():
        try: return pd.read_csv(JOURNAL_PATH)
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

def save_journal(df: pd.DataFrame) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(JOURNAL_PATH, index=False)

def append_journal(row: dict) -> bool:
    df = load_journal()
    if not df.empty and "id" in df.columns and row["id"] in set(df["id"].astype(str)):
        return False
    save_journal(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True

def evaluate_journal(current_prices: dict[str, float]) -> pd.DataFrame:
    df = load_journal()
    if df.empty: return df
    df = df.copy()
    for idx, row in df.iterrows():
        sym, entry, decision = str(row.get("symbol", "")), float(row.get("price", np.nan)), str(row.get("decision", ""))
        cur = current_prices.get(sym, np.nan)
        if not np.isfinite(entry) or not np.isfinite(cur): continue
        ret = (cur / entry - 1) * 100
        if decision == "VENDER": ret = -ret
        elif decision not in {"COMPRAR", "VENDER"}: ret = np.nan
        df.loc[idx, "current_price"] = cur
        df.loc[idx, "open_return_pct"] = ret
        df.loc[idx, "evaluated_at"] = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
    save_journal(df)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Copiloto IA + registro operacional em tempo real
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _write_csv_safe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_decision_log() -> pd.DataFrame:
    return _read_csv_safe(DECISION_LOG_PATH)


def append_decision_log(row: dict) -> bool:
    df = load_decision_log()
    if not df.empty and "id" in df.columns and row.get("id") in set(df["id"].astype(str)):
        return False
    _write_csv_safe(DECISION_LOG_PATH, pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True


def load_trade_log() -> pd.DataFrame:
    return _read_csv_safe(TRADE_LOG_PATH)


def save_trade_log(df: pd.DataFrame) -> None:
    _write_csv_safe(TRADE_LOG_PATH, df)


def _trade_id(symbol: str, side: str, ts: str, entry: float) -> str:
    raw = f"{symbol}|{side}|{ts}|{entry:.6f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def pnl_cash(side: str, entry: float, mark: float, qty: float, point_value: float, fees: float = 0.0) -> float:
    if not all(np.isfinite(x) for x in [entry, mark, qty, point_value, fees]):
        return np.nan
    direction = 1.0 if side == "COMPRAR" else -1.0 if side == "VENDER" else 0.0
    return direction * (mark - entry) * qty * point_value - fees


def append_open_trade(row: dict) -> bool:
    df = load_trade_log()
    if not df.empty and "id" in df.columns and row.get("id") in set(df["id"].astype(str)):
        return False
    _write_csv_safe(TRADE_LOG_PATH, pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True


def update_trade_row(trade_id_: str, updates: dict) -> bool:
    df = load_trade_log()
    if df.empty or "id" not in df.columns:
        return False
    mask = df["id"].astype(str) == str(trade_id_)
    if not mask.any():
        return False
    for k, v in updates.items():
        df.loc[mask, k] = v
    save_trade_log(df)
    return True


def open_trades(symbol: str | None = None) -> pd.DataFrame:
    df = load_trade_log()
    if df.empty:
        return df
    if "status" not in df.columns:
        return pd.DataFrame()
    out = df[df["status"].astype(str) == "ABERTA"].copy()
    if symbol:
        out = out[out["symbol"].astype(str) == symbol]
    return out


def decision_snapshot(r: CoachReport, ml: MLReport, decision: str, score: float, reason: str, plan: RiskPlan, audit: DecisionAudit | None = None, strategy: TimeframeStrategy | None = None) -> dict:
    ml_prob = ml.calibrated_probability if np.isfinite(ml.calibrated_probability) else ml.probability
    return {
        "timestamp": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "symbol": r.symbol,
        "price": r.price,
        "suggestion": decision,
        "score": float(score),
        "reason": reason,
        "trend": f"{r.trend} {r.trend_strength}",
        "reversal_risk": r.reversal_risk,
        "session": r.session.name,
        "ml_prediction": ml.prediction,
        "ml_probability": float(ml_prob) if np.isfinite(ml_prob) else np.nan,
        "ml_quality": float(ml.model_quality) if np.isfinite(ml.model_quality) else np.nan,
        "stop": plan.stop,
        "target": plan.target,
        "rr": plan.rr,
        "audit_reason": audit.rationale if audit else "",
        "timeframe_minutes": strategy.minutes if strategy else np.nan,
        "timeframe_label": strategy.label if strategy else "",
        "timeframe_style": strategy.style if strategy else "",
        "timeframe_setup": strategy.preferred_setups if strategy else "",
    }


def local_copilot_reply(question: str, snap: dict, trade_context: str = "") -> str:
    decision = str(snap.get("suggestion", "AGUARDAR"))
    score = float(snap.get("score", 0.0) or 0.0)
    risk = str(snap.get("reversal_risk", ""))
    session = str(snap.get("session", ""))
    mlq = snap.get("ml_quality", np.nan)
    mlp = snap.get("ml_probability", np.nan)
    warnings_: list[str] = []
    if risk == "ALTO":
        warnings_.append("risco de reversão alto")
    if session in {"PAUSA FUTUROS", "FECHADO (FDS FUTUROS)", "ALMOÇO NY", "OVERNIGHT FUTUROS"}:
        warnings_.append(f"sessão {session.lower()} exige cautela")
    if not np.isfinite(mlq) or float(mlq) < 0.52:
        warnings_.append("ML ainda não tem qualidade temporal forte")
    if decision in {"COMPRAR", "VENDER"} and score >= 1.55 and not warnings_:
        stance = "O sinal é operacionalmente aceitável, desde que a entrada respeite stop, tamanho e invalidação."
    elif decision in {"COMPRAR", "VENDER"}:
        stance = "O sinal existe, mas eu trataria como entrada condicional, não como entrada automática."
    elif decision == "SAIR · PROTEGER":
        stance = "A prioridade é defesa: reduzir exposição, mover stop ou encerrar posição vulnerável."
    else:
        stance = "A leitura favorece esperar; neste momento, o melhor trade pode ser não operar."
    warn_txt = "; ".join(warnings_) if warnings_ else "sem bloqueio crítico aparente"
    ml_txt = f"ML p={float(mlp):.0%}, qualidade={float(mlq):.2f}" if np.isfinite(mlp) and np.isfinite(mlq) else "ML sem confirmação robusta"
    return (
        f"**Copiloto local**\n\n"
        f"- Leitura: **{stance}**\n"
        f"- Base: sugestão `{decision}`, score `{score:.2f}`, tendência `{snap.get('trend', '')}`, sessão `{session}`.\n"
        f"- Checagem ML: {ml_txt}.\n"
        f"- Atenção: {warn_txt}.\n"
        f"- Para decidir agora: aceite apenas se o preço real, seu tamanho e o stop forem compatíveis com o risco financeiro; se houver dúvida, registre a recusa/espera e aguarde nova confluência.\n"
        f"- Sua pergunta: {question.strip() or 'sem pergunta específica'}\n"
        f"{trade_context}"
    )


def call_remote_copilot(provider: str, model: str, api_key: str, base_url: str, question: str, snap: dict, trade_context: str) -> tuple[bool, str]:
    system = (
        "Você é um copiloto de trading conservador dentro de um app Streamlit. "
        "Não promete lucro, não dá garantias e não incentiva alavancagem. "
        "Responda em português, em até 7 bullets curtos. "
        "Foque em: aceitar/recusar/esperar, janela temporal da operação, risco, invalidação, tamanho da posição e registro disciplinado."
    )
    user = json.dumps({"pergunta": question, "snapshot": snap, "posicoes_abertas": trade_context}, ensure_ascii=False, default=str)
    try:
        if provider == "Ollama local":
            url = (base_url or "http://localhost:11434").rstrip("/") + "/api/chat"
            payload = {"model": model or "llama3.1", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "stream": False}
            resp = requests.post(url, json=payload, timeout=60)
            if not resp.ok:
                return False, resp.text
            return True, resp.json().get("message", {}).get("content", "Sem resposta.")
        if provider in {"OpenAI compatível", "Groq"}:
            if provider == "Groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                key = api_key or os.getenv("GROQ_API_KEY", "")
                model = model or "llama-3.3-70b-versatile"
            else:
                url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/") + "/chat/completions"
                key = api_key or os.getenv("OPENAI_API_KEY", "")
                model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            if not key:
                return False, "Chave de API ausente. Use variável de ambiente ou campo de chave na interface."
            payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.2, "max_tokens": 700}
            resp = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=60)
            if not resp.ok:
                return False, resp.text
            return True, resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return False, f"Falha no copiloto remoto: {exc}"
    return False, "Provedor não reconhecido."


def copilot_reply(provider: str, model: str, api_key: str, base_url: str, question: str, snap: dict, trade_context: str) -> str:
    if provider == "Heurístico local" or requests is None:
        return local_copilot_reply(question, snap, trade_context)
    ok, ans = call_remote_copilot(provider, model, api_key, base_url, question, snap, trade_context)
    if ok:
        return ans
    fallback = local_copilot_reply(question, snap, trade_context)
    return f"⚠️ Copiloto remoto indisponível: {ans}\n\n{fallback}"


def render_copilot_and_realtime_log(
    r: CoachReport,
    ml: MLReport,
    decision: str,
    score: float,
    reason: str,
    plan: RiskPlan,
    audit: DecisionAudit,
    ai_provider: str,
    ai_model: str,
    ai_api_key: str,
    ai_base_url: str,
    strategy: TimeframeStrategy,
) -> None:
    snap = decision_snapshot(r, ml, decision, score, reason, plan, audit, strategy)
    sym = r.symbol
    open_df = open_trades(sym)
    open_txt = ""
    if not open_df.empty:
        cols = ["id", "side", "entry_price", "quantity", "point_value", "mark_price", "unrealized_pnl", "stop", "target", "note"]
        open_txt = open_df[[c for c in cols if c in open_df.columns]].tail(5).to_json(orient="records", force_ascii=False)

    with st.expander(f"Copiloto IA + registro real · {sym}", expanded=False):
        top = st.columns([1.1, 1.1, 1.1, 1.1])
        operator_choice = top[0].selectbox("Minha decisão", ["ACEITAR SINAL", "RECUSAR", "ESPERAR", "SAIR/PROTEGER"], key=f"choice_{sym}")
        actual_action = top[1].selectbox("Ação real", [decision, "COMPRAR", "VENDER", "AGUARDAR", "SAIR · PROTEGER"], key=f"actual_action_{sym}")
        real_price = top[2].number_input("Preço real", value=float(r.price), step=float(max(r.atr / 10, 0.01)), format="%.5f", key=f"real_price_{sym}")
        confidence_user = top[3].slider("Minha convicção", 0, 100, 50, key=f"user_conf_{sym}")
        note_dec = st.text_area("Nota rápida da decisão", key=f"dec_note_{sym}", height=70, placeholder="Ex.: aceitei porque rompeu resistência; recusei porque estava fora da janela; esperei por falta de volume.")
        dc = st.columns([1, 1, 2])
        if dc[0].button("Registrar decisão", key=f"register_decision_{sym}"):
            ts = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
            row = snap | {
                "id": signal_id(sym, ts, actual_action, float(real_price)),
                "operator_choice": operator_choice,
                "actual_action": actual_action,
                "real_price": float(real_price),
                "user_confidence": int(confidence_user),
                "operator_note": note_dec,
            }
            append_decision_log(row)
            st.success("Decisão registrada.")
        if dc[1].button("Perguntar à IA", key=f"ask_ai_button_{sym}"):
            st.session_state[f"ai_pending_{sym}"] = True
        question = st.text_area("Pergunta ao copiloto", key=f"ai_question_{sym}", height=80, placeholder="Ex.: aceito essa venda agora ou espero pullback? O que invalidaria esse sinal?")
        if st.session_state.get(f"ai_pending_{sym}"):
            with st.spinner("Copiloto analisando contexto operacional..."):
                ans = copilot_reply(ai_provider, ai_model, ai_api_key, ai_base_url, question, snap, open_txt)
            st.session_state.setdefault(f"ai_history_{sym}", []).append({"q": question, "a": ans, "ts": datetime.now(LOCAL_TZ).isoformat(timespec="seconds")})
            st.session_state[f"ai_pending_{sym}"] = False
        hist = st.session_state.get(f"ai_history_{sym}", [])
        for item in hist[-3:]:
            st.markdown(f"**Você ({item['ts']})**: {item['q'] or 'análise do sinal atual'}")
            st.markdown(item["a"])
            st.divider()

        st.markdown("##### Abrir posição real/manual")
        if actual_action in {"COMPRAR", "VENDER"}:
            oc = st.columns(6)
            qty = oc[0].number_input("Qtd.", min_value=0.0, value=1.0, step=1.0, key=f"qty_{sym}")
            point_value = oc[1].number_input("Valor por ponto", min_value=0.0, value=1.0, step=1.0, key=f"pv_{sym}")
            fees = oc[2].number_input("Custos", min_value=0.0, value=0.0, step=1.0, key=f"fees_{sym}")
            stop_real = oc[3].number_input("Stop", value=float(plan.stop) if np.isfinite(plan.stop) else float(real_price), step=float(max(r.atr / 10, 0.01)), format="%.5f", key=f"stop_real_{sym}")
            target_real = oc[4].number_input("Alvo", value=float(plan.target) if np.isfinite(plan.target) else float(real_price), step=float(max(r.atr / 10, 0.01)), format="%.5f", key=f"target_real_{sym}")
            oc[5].metric("P&L se marcar agora", f"{pnl_cash(actual_action, float(real_price), float(r.price), float(qty), float(point_value), float(fees)):+.2f}")
            if st.button("Abrir posição no registro", key=f"open_trade_{sym}"):
                ts = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
                tid = _trade_id(sym, actual_action, ts, float(real_price))
                row = {
                    "id": tid, "status": "ABERTA", "timestamp_open": ts, "timestamp_close": "",
                    "symbol": sym, "side": actual_action, "suggestion": decision, "operator_choice": operator_choice,
                    "entry_price": float(real_price), "quantity": float(qty), "point_value": float(point_value), "fees": float(fees),
                    "stop": float(stop_real), "target": float(target_real), "mark_price": float(r.price),
                    "unrealized_pnl": pnl_cash(actual_action, float(real_price), float(r.price), float(qty), float(point_value), float(fees)),
                    "exit_price": np.nan, "realized_pnl": np.nan, "score": float(score), "ml_quality": snap.get("ml_quality", np.nan),
                    "session": r.session.name, "trend": snap.get("trend", ""), "timeframe_minutes": strategy.minutes, "timeframe_label": strategy.label, "note": note_dec,
                }
                append_open_trade(row)
                st.success(f"Posição aberta no registro: {tid}")
        else:
            st.caption("Para abrir posição manual, selecione COMPRAR ou VENDER em 'Ação real'.")

        st.markdown("##### Posições abertas e P&L em tempo real")
        odf = open_trades(sym)
        if odf.empty:
            st.caption("Nenhuma posição aberta para este ativo.")
        else:
            for _, tr in odf.tail(5).iterrows():
                tid = str(tr.get("id"))
                side = str(tr.get("side"))
                entry = float(tr.get("entry_price", np.nan))
                qty0 = float(tr.get("quantity", 0.0))
                pv0 = float(tr.get("point_value", 1.0))
                fees0 = float(tr.get("fees", 0.0))
                with st.container(border=True):
                    st.write(f"**{tid} · {side} · entrada {fmt_price(entry, r.asset_type)}**")
                    tc = st.columns(5)
                    mark = tc[0].number_input("Preço atual/saída", value=float(tr.get("mark_price", r.price)) if np.isfinite(float(tr.get("mark_price", r.price))) else float(r.price), step=float(max(r.atr / 10, 0.01)), format="%.5f", key=f"mark_{tid}")
                    add_fees = tc[1].number_input("Custos totais", value=fees0, min_value=0.0, step=1.0, key=f"fees_mark_{tid}")
                    pnl = pnl_cash(side, entry, float(mark), qty0, pv0, float(add_fees))
                    tc[2].metric("P&L", f"{pnl:+.2f}")
                    tc[3].metric("Pts", f"{((float(mark)-entry) * (1 if side=='COMPRAR' else -1)):+.2f}")
                    tc[4].metric("Status", "ABERTA")
                    uc = st.columns([1, 1, 2])
                    if uc[0].button("Atualizar P&L", key=f"upd_{tid}"):
                        update_trade_row(tid, {"mark_price": float(mark), "fees": float(add_fees), "unrealized_pnl": float(pnl), "updated_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds")})
                        st.success("P&L atualizado.")
                    if uc[1].button("Fechar posição", key=f"close_{tid}"):
                        update_trade_row(tid, {"status": "FECHADA", "timestamp_close": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"), "exit_price": float(mark), "fees": float(add_fees), "realized_pnl": float(pnl), "unrealized_pnl": np.nan})
                        st.success("Posição fechada no registro.")
                    close_note = uc[2].text_input("Comentário", key=f"trade_note_{tid}")
                    if close_note:
                        update_trade_row(tid, {"note": close_note})

def _load_alerts_sent() -> dict:
    if ALERTS_SENT_PATH.exists():
        try: return json.loads(ALERTS_SENT_PATH.read_text())
        except Exception: return {}
    return {}

def _save_alerts_sent(state: dict) -> None:
    try:
        ALERTS_SENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERTS_SENT_PATH.write_text(json.dumps(state))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Gates persistentes — V7.4
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Converte objetos numpy/pandas em tipos serializáveis para JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


def gates_signature(symbols: list[str] | tuple[str, ...], period: str, interval: str, horizon_bars: int,
                    model_kind: str, stop_atr: float, target_atr: float, strictness: float,
                    cost_bps: float = 2.0, slippage_bps: float = 1.0) -> dict:
    """Assinatura operacional usada para saber se os gates salvos ainda combinam com a configuração atual."""
    return {
        "symbols": list(symbols),
        "period": str(period),
        "interval": str(interval),
        "horizon_bars": int(horizon_bars),
        "model_kind": str(model_kind),
        "stop_atr": round(float(stop_atr), 4),
        "target_atr": round(float(target_atr), 4),
        "strictness": round(float(strictness), 4),
        "cost_bps": round(float(cost_bps), 4),
        "slippage_bps": round(float(slippage_bps), 4),
        "version": APP_VERSION,
    }


def signature_match(saved: dict, current: dict) -> bool:
    """Comparação tolerante: exige os mesmos ativos e parâmetros principais."""
    try:
        return (
            set(saved.get("symbols", [])) == set(current.get("symbols", []))
            and saved.get("period") == current.get("period")
            and saved.get("interval") == current.get("interval")
            and int(saved.get("horizon_bars", -1)) == int(current.get("horizon_bars", -2))
            and saved.get("model_kind") == current.get("model_kind")
            and abs(float(saved.get("stop_atr", -999)) - float(current.get("stop_atr", 999))) < 1e-6
            and abs(float(saved.get("target_atr", -999)) - float(current.get("target_atr", 999))) < 1e-6
        )
    except Exception:
        return False


def load_gates_cache() -> dict:
    if not GATES_CACHE_PATH.exists():
        return {"available": False, "reason": "sem arquivo de gates salvo"}
    try:
        payload = json.loads(GATES_CACHE_PATH.read_text())
        gates = payload.get("gates", {})
        if not isinstance(gates, dict):
            return {"available": False, "reason": "arquivo de gates inválido"}
        gates.setdefault("available", False)
        gates["_cache_meta"] = {
            "created_at": payload.get("created_at", ""),
            "signature": payload.get("signature", {}),
            "path": str(GATES_CACHE_PATH),
        }
        return gates
    except Exception as exc:
        return {"available": False, "reason": f"falha ao carregar gates salvos: {exc}"}


def save_gates_cache(gates: dict, signature: dict) -> tuple[bool, str]:
    try:
        GATES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
            "signature": _json_safe(signature),
            "gates": _json_safe(gates),
        }
        GATES_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return True, str(GATES_CACHE_PATH)
    except Exception as exc:
        return False, str(exc)


def delete_gates_cache() -> tuple[bool, str]:
    try:
        if GATES_CACHE_PATH.exists():
            GATES_CACHE_PATH.unlink()
        return True, "gates salvos apagados"
    except Exception as exc:
        return False, str(exc)


def gates_age_label(gates: dict) -> str:
    meta = gates.get("_cache_meta", {}) if isinstance(gates, dict) else {}
    raw = meta.get("created_at", "")
    if not raw:
        return "Gates não calibrados ainda."
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        delta = datetime.now(LOCAL_TZ) - dt.astimezone(LOCAL_TZ)
        mins = max(int(delta.total_seconds() // 60), 0)
        if mins < 60:
            return f"Gates calibrados há {mins} min."
        hours, rem = divmod(mins, 60)
        if hours < 48:
            return f"Gates calibrados há {hours}h{rem:02d}min."
        days, h = divmod(hours, 24)
        return f"Gates calibrados há {days}d{h:02d}h."
    except Exception:
        return "Gates salvos, mas sem data legível."

def send_telegram_alert(message: str) -> tuple[bool, str]:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram não configurado."
    if requests is None:
        return False, "requests indisponível."
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        return (True, "Enviado.") if resp.ok else (False, f"Erro Telegram: {resp.text}")
    except Exception as exc:
        return False, f"Falha de rede: {exc}"


def _telegram_escape(text: Any) -> str:
    """Escapa caracteres problemáticos em mensagens Telegram Markdown simples."""
    s = str(text)
    for ch in ["`", "*", "_", "["]:
        s = s.replace(ch, "")
    return s


def best_timeframe_for_signal(
    r: CoachReport,
    ml: MLReport,
    decision: str,
    selected_strategy: TimeframeStrategy,
) -> tuple[TimeframeStrategy, str]:
    """Escolhe uma janela temporal razoável para o alerta.

    Esta é uma recomendação prudencial, não uma otimização ex-post. Ela combina:
    tipo de ativo, liquidez da sessão, força da tendência, risco de reversão,
    estabilidade do par FX e qualidade do ML. O objetivo é evitar que um alerta
    de alta convicção sugira o mesmo manejo para scalp de 1 minuto e trade de 4h.
    """
    if decision not in {"COMPRAR", "VENDER"}:
        return selected_strategy, "sem entrada direcional; manter a janela selecionada para observação"

    ml_quality = ml.model_quality if ml.available and np.isfinite(ml.model_quality) else 0.45
    tier = forex_stability_label(r.symbol) if r.asset_type == "forex" else "índice"
    scores: dict[int, float] = {}

    for minutes in TIMEFRAME_WINDOWS:
        stg = timeframe_strategy(minutes)
        score = 0.0

        # A própria janela escolhida pelo operador recebe um pequeno bônus,
        # mas não domina a recomendação se o contexto pedir outro manejo.
        if stg.minutes == selected_strategy.minutes:
            score += 0.20

        # Liquidez e sessão: índices favorecem 15-60 min nas janelas boas;
        # overnight/pós-mercado pedem mais paciência e confirmação.
        if r.asset_type == "index":
            if r.session.name in {"MANHÃ", "TARDE"}:
                score += {1: -0.10, 5: 0.10, 15: 0.55, 30: 0.65, 60: 0.45, 120: 0.20, 240: 0.05}.get(minutes, 0)
            elif r.session.name in {"PRÉ-MARKET FUTUROS", "OVERNIGHT FUTUROS", "PÓS-MERCADO FUTUROS"}:
                score += {1: -0.60, 5: -0.25, 15: 0.10, 30: 0.35, 60: 0.50, 120: 0.55, 240: 0.35}.get(minutes, 0)
            else:
                score += -0.50
        else:
            # Pares estáveis tendem a funcionar melhor em 15-60 min;
            # janelas muito curtas sofrem mais com spread/ruído.
            if tier == "estável":
                score += {1: -0.45, 5: 0.05, 15: 0.55, 30: 0.65, 60: 0.55, 120: 0.20, 240: 0.05}.get(minutes, 0)
            elif tier == "intermediário":
                score += {1: -0.35, 5: 0.05, 15: 0.40, 30: 0.55, 60: 0.50, 120: 0.30, 240: 0.10}.get(minutes, 0)
            else:
                score += {1: -0.50, 5: -0.05, 15: 0.25, 30: 0.45, 60: 0.55, 120: 0.45, 240: 0.20}.get(minutes, 0)

        # Tendência forte aceita janelas maiores; lateralidade favorece janelas
        # táticas, mas não micro-scalp automático.
        if r.trend_strength == "FORTE":
            score += {1: -0.10, 5: 0.05, 15: 0.25, 30: 0.35, 60: 0.50, 120: 0.45, 240: 0.30}.get(minutes, 0)
        elif r.trend == "LATERAL":
            score += {1: -0.20, 5: 0.20, 15: 0.35, 30: 0.25, 60: 0.05, 120: -0.10, 240: -0.20}.get(minutes, 0)

        if r.reversal_risk == "MÉDIO":
            score += {1: -0.15, 5: 0.15, 15: 0.20, 30: 0.10, 60: -0.05, 120: -0.15, 240: -0.25}.get(minutes, 0)
        elif r.reversal_risk == "BAIXO":
            score += {15: 0.10, 30: 0.15, 60: 0.20, 120: 0.15, 240: 0.10}.get(minutes, 0)

        if ml_quality >= 0.58:
            score += {15: 0.05, 30: 0.10, 60: 0.15, 120: 0.10}.get(minutes, 0)
        elif ml_quality < 0.50:
            score += {1: -0.20, 5: -0.10, 15: 0.05, 30: 0.10, 60: 0.10, 120: 0.05}.get(minutes, 0)

        # Dados problemáticos: nunca recomendar micro-janelas.
        if r.data_health.status != "OK" and minutes in {1, 5}:
            score -= 0.60

        scores[minutes] = float(score)

    best_minutes = max(scores, key=scores.get)
    best = timeframe_strategy(best_minutes)
    reason = (
        f"{best.label}: melhor equilíbrio entre {r.session.name}, "
        f"tendência {r.trend} {r.trend_strength}, risco {r.reversal_risk}, "
        f"perfil {tier} e qualidade ML {ml_quality:.2f}"
    )
    return best, reason


def telegram_alert_message(
    r: CoachReport,
    ml: MLReport,
    decision: str,
    score: float,
    plan: RiskPlan,
    strategy: TimeframeStrategy,
    directed_goal: DirectedGoal | None = None,
    audit: DecisionAudit | None = None,
) -> str:
    ml_prob = ml.calibrated_probability if np.isfinite(ml.calibrated_probability) else ml.probability
    best_tf, best_tf_reason = best_timeframe_for_signal(r, ml, decision, strategy)
    emoji = "🟢" if decision == "COMPRAR" else "🔴" if decision == "VENDER" else "🟡"
    pair_label = "Par" if r.asset_type == "forex" else "Ativo"
    rr_txt = f"{plan.rr:.2f}" if np.isfinite(plan.rr) else "—"
    stop_txt = fmt_price(plan.stop, r.asset_type) if np.isfinite(plan.stop) else "—"
    target_txt = fmt_price(plan.target, r.asset_type) if np.isfinite(plan.target) else "—"
    units_txt = f"{plan.suggested_units:.2f}" if np.isfinite(plan.suggested_units) else "—"
    tier = forex_stability_label(r.symbol) if r.asset_type == "forex" else "índice"

    directed_lines = ""
    if directed_goal is not None and directed_goal.enabled:
        directed_lines = (
            f"\n🎯 *Modo dirigido*\n"
            f"Capital: `{directed_goal.capital_cash:.2f}` · Meta: `{directed_goal.target_profit_cash:.2f}` · "
            f"Perda máx.: `{directed_goal.max_loss_cash:.2f}`\n"
            f"R/R mínimo: `{directed_goal.min_rr:.2f}` · Máx. operações: `{directed_goal.max_trades}`"
        )

    blockers = ""
    if audit is not None and audit.blockers:
        blockers = "\n⚠️ Bloqueios/rebaixamentos: " + _telegram_escape("; ".join(audit.blockers[:3]))

    return (
        f"{emoji} *F-Coach — ALERTA {decision}*\n"
        f"{pair_label}: `{_telegram_escape(r.symbol)}` · Perfil: `{_telegram_escape(tier)}`\n"
        f"Preço ref.: `{fmt_price(r.price, r.asset_type)}` · Sessão: `{_telegram_escape(r.session.name)}`\n"
        f"\n📌 *Risco*\n"
        f"Reversão: `{r.reversal_risk}` · ATR: `{r.atr_pct:.2f}%` · Dados: `{_telegram_escape(r.data_health.status)}`\n"
        f"Score: `{score:.2f}` · ML: `{_telegram_escape(ml.prediction)}` p=`{ml_prob:.0%}` qualidade=`{ml.model_quality:.2f}`\n"
        f"\n⏱️ *Melhor timeframe*\n"
        f"`{_telegram_escape(best_tf.label)}`\n"
        f"Motivo: {_telegram_escape(best_tf_reason)}\n"
        f"Janela selecionada no app: `{_telegram_escape(strategy.label)}`\n"
        f"\n🧭 *Plano já calculado*\n"
        f"Entrada ref.: `{fmt_price(plan.entry, r.asset_type)}` · Stop: `{stop_txt}` · Alvo: `{target_txt}`\n"
        f"R/R: `{rr_txt}` · Unidades pelo risco: `{units_txt}` · Risco-base: `{plan.risk_cash:.2f}`\n"
        f"Invalidação: {_telegram_escape(plan.invalidation)}"
        f"{directed_lines}"
        f"{blockers}"
        f"\n\nConfirme preço real, spread e stop na corretora antes de registrar a operação."
    )


def maybe_send_high_conviction_alert(
    r: CoachReport,
    ml: MLReport,
    decision: str,
    score: float,
    plan: RiskPlan,
    strategy: TimeframeStrategy,
    directed_goal: DirectedGoal | None = None,
    audit: DecisionAudit | None = None,
) -> Optional[str]:
    if decision not in {"COMPRAR", "VENDER"}: return None
    if score < HIGH_CONVICTION_MIN_SCORE: return None
    if not ml.available or not np.isfinite(ml.model_quality) or ml.model_quality < HIGH_CONVICTION_MIN_ML_QUALITY: return None
    ml_prob = ml.calibrated_probability if np.isfinite(ml.calibrated_probability) else ml.probability
    if not np.isfinite(ml_prob) or ml_prob < HIGH_CONVICTION_MIN_ML_PROB: return None
    if ml.prediction != decision: return None
    if ml.conformal_available:
        cset = set(str(x) for x in ml.conformal_set)
        opposite = "VENDER" if decision == "COMPRAR" else "COMPRAR"
        if decision not in cset or opposite in cset or len(cset) >= 3:
            return None
    if r.asset_type == "index" and r.session.name not in HIGH_CONVICTION_GOOD_SESSIONS: return None
    if r.reversal_risk == "ALTO": return None
    best_tf, _ = best_timeframe_for_signal(r, ml, decision, strategy)
    key = f"{r.symbol}|{decision}|tf{best_tf.minutes}|{round(r.price / max(r.atr * 0.5, 1e-6))}"
    state = _load_alerts_sent()
    if state.get(key): return None
    msg = telegram_alert_message(r, ml, decision, score, plan, strategy, directed_goal, audit)
    ok, status = send_telegram_alert(msg)
    state[key] = {
        "ts": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "symbol": r.symbol,
        "decision": decision,
        "best_timeframe_minutes": best_tf.minutes,
        "price": r.price,
        "score": score,
        "sent": ok,
        "status": status,
    }
    _save_alerts_sent(state)
    return f"Alerta {'enviado' if ok else 'falhou'}: {status}"


# ─────────────────────────────────────────────────────────────────────────────
# Modo dirigido: capital, meta e roteiros condicionais
# ─────────────────────────────────────────────────────────────────────────────

def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def directed_goal_status(goal: DirectedGoal, realized_pnl: float = 0.0, open_pnl: float = 0.0) -> dict[str, float | str]:
    target = max(float(goal.target_profit_cash), 1e-9)
    realized = _finite_float(realized_pnl)
    open_ = _finite_float(open_pnl)
    total = realized + open_
    remaining = max(goal.target_profit_cash - total, 0.0)
    loss_used = max(-total, 0.0)
    loss_remaining = max(goal.max_loss_cash - loss_used, 0.0)
    return {
        "pnl_realizado": realized,
        "pnl_aberto": open_,
        "pnl_total": total,
        "meta": float(goal.target_profit_cash),
        "progresso_meta": min(max(total / target, -1.0), 2.0),
        "faltante_meta": remaining,
        "perda_maxima": float(goal.max_loss_cash),
        "folga_perda": loss_remaining,
        "estado": "META ATINGIDA" if total >= goal.target_profit_cash else "LIMITE DE PERDA" if loss_used >= goal.max_loss_cash else "EM ANDAMENTO",
    }


def summarize_directed_goal(goal: DirectedGoal) -> str:
    risk_pct = 100 * goal.max_loss_cash / max(goal.capital_cash, 1e-9)
    target_pct = 100 * goal.target_profit_cash / max(goal.capital_cash, 1e-9)
    return (
        f"Capital alocado: {goal.capital_cash:.2f}. Meta de lucro: {goal.target_profit_cash:.2f} "
        f"({target_pct:.2f}% do capital). Perda máxima planejada: {goal.max_loss_cash:.2f} "
        f"({risk_pct:.2f}%). Máximo de operações roteirizadas: {goal.max_trades}."
    )


def build_directed_routes(
    packs: list[tuple[CoachReport, MLReport, str, float, str, RiskPlan, DecisionAudit]],
    goal: DirectedGoal,
    strategy: TimeframeStrategy,
) -> list[DirectedRoute]:
    """Converte sinais calibrados em roteiros condicionais compatíveis com a meta.

    A função não executa ordens. Ela calcula se a oportunidade é compatível com o
    orçamento de risco e com a meta definida pelo operador. O roteiro é sempre
    condicional: entrada apenas se o preço e o contexto continuarem válidos.
    """
    if not goal.enabled:
        return []
    point_value = max(float(goal.point_value), 1e-9)
    max_trades = max(int(goal.max_trades), 1)
    risk_per_trade = max(float(goal.max_loss_cash) / max_trades, 0.0)
    routes: list[DirectedRoute] = []

    for r, ml, decision, score, reason, plan, audit in packs:
        if decision not in {"COMPRAR", "VENDER"}:
            continue
        if not np.isfinite(plan.risk_points) or plan.risk_points <= 0:
            continue
        if r.data_health.status == "RUIM":
            continue
        if r.session.directional_permission < strategy.min_session_permission:
            continue
        if r.reversal_risk == "ALTO":
            continue

        raw_units = risk_per_trade / max(plan.risk_points * point_value, 1e-9)
        units = min(float(goal.max_units), math.floor(raw_units))
        if goal.max_units < 1:
            units = min(float(goal.max_units), raw_units)
        if units <= 0:
            continue

        risk_cash = units * plan.risk_points * point_value
        reward_cash = units * plan.reward_points * point_value
        rr = reward_cash / risk_cash if risk_cash > 0 else np.nan
        if np.isfinite(goal.min_rr) and np.isfinite(rr) and rr < goal.min_rr:
            # Mantém apenas se o usuário permitir alvos parciais e a cobertura for expressiva.
            if not goal.allow_partial_targets or reward_cash < goal.target_profit_cash * 0.35:
                continue

        ml_quality = ml.model_quality if ml.available and np.isfinite(ml.model_quality) else 0.40
        health_bonus = 0.15 if r.data_health.status == "OK" else 0.0
        route_score = float(score) + 0.35 * float(ml_quality) + health_bonus + 0.10 * min(max(rr, 0), 3)
        coverage = reward_cash / max(goal.target_profit_cash, 1e-9)
        action_verb = "comprar" if decision == "COMPRAR" else "vender"
        invalidation = plan.invalidation.rstrip(".")
        status = "roteiro principal" if coverage >= 0.70 else "roteiro parcial" if coverage >= 0.30 else "roteiro pequeno"
        roteiro = [
            f"Preparar {action_verb} {r.symbol} somente enquanto a decisão calibrada permanecer {decision} e o preço estiver próximo de {fmt_price(plan.entry, r.asset_type)}.",
            f"Entrada de referência: {fmt_price(plan.entry, r.asset_type)}; stop obrigatório em {fmt_price(plan.stop, r.asset_type)}; alvo em {fmt_price(plan.target, r.asset_type)}.",
            f"Tamanho sugerido: {units:.2f} unidade(s), considerando valor por ponto {point_value:.2f}; risco estimado {risk_cash:.2f}; ganho potencial {reward_cash:.2f}; R/R {rr:.2f}.",
            f"Não entrar se o candle já estiver esticado, se a sessão mudar para pausa/baixa liquidez ou se ocorrer invalidação: {invalidation}.",
            f"Gestão pela janela {strategy.label}: {strategy.management}",
        ]
        routes.append(DirectedRoute(
            symbol=r.symbol,
            action=decision,
            priority=0,
            route_score=route_score,
            entry=plan.entry,
            stop=plan.stop,
            target=plan.target,
            units=units,
            risk_cash=risk_cash,
            reward_cash=reward_cash,
            rr=rr,
            target_coverage=coverage,
            status=status,
            roteiro=roteiro,
        ))

    routes.sort(key=lambda x: (x.route_score, x.target_coverage, x.rr), reverse=True)
    routes = routes[: max(int(goal.max_trades), 1)]
    for i, route in enumerate(routes, start=1):
        route.priority = i
    return routes


def render_directed_mode_panel(
    packs: list[tuple[CoachReport, MLReport, str, float, str, RiskPlan, DecisionAudit]],
    goal: DirectedGoal,
    strategy: TimeframeStrategy,
) -> None:
    if not goal.enabled:
        return
    tlog = load_trade_log()
    realized = 0.0
    open_pnl = 0.0
    if not tlog.empty:
        if "realized_pnl" in tlog.columns:
            realized = float(pd.to_numeric(tlog.get("realized_pnl"), errors="coerce").fillna(0).sum())
        if "unrealized_pnl" in tlog.columns:
            open_mask = tlog.get("status", pd.Series(dtype=str)).astype(str) == "ABERTA"
            open_pnl = float(pd.to_numeric(tlog.loc[open_mask, "unrealized_pnl"], errors="coerce").fillna(0).sum()) if open_mask.any() else 0.0
    status = directed_goal_status(goal, realized, open_pnl)
    routes = build_directed_routes(packs, goal, strategy)

    with st.container(border=True):
        st.subheader("Modo dirigido · roteiro por meta")
        st.caption("Roteiros condicionais: o app não executa ordens e deve ser usado com stop, validação da corretora e confirmação do operador.")
        cols = st.columns(5)
        cols[0].metric("Capital", f"{goal.capital_cash:.2f}")
        cols[1].metric("Meta", f"{goal.target_profit_cash:.2f}", f"faltam {status['faltante_meta']:.2f}")
        cols[2].metric("Perda máxima", f"{goal.max_loss_cash:.2f}", f"folga {status['folga_perda']:.2f}")
        cols[3].metric("P&L total", f"{status['pnl_total']:.2f}", f"{status['progresso_meta']:.0%} da meta")
        cols[4].metric("Estado", str(status["estado"]))

        if status["estado"] == "META ATINGIDA":
            st.success("Meta atingida. Roteiro prudente: parar, proteger lucro e revisar diário antes de nova operação.")
            return
        if status["estado"] == "LIMITE DE PERDA":
            st.error("Limite de perda atingido. Roteiro prudente: encerrar novas entradas e revisar o plano.")
            return
        if not routes:
            st.warning("Nenhuma operação compatível com a meta e o orçamento de risco neste momento. Roteiro: aguardar nova confluência.")
            return

        table = pd.DataFrame([{
            "prioridade": rt.priority,
            "ativo": rt.symbol,
            "ação": rt.action,
            "status": rt.status,
            "score_rota": round(rt.route_score, 2),
            "entrada": round(rt.entry, 5),
            "stop": round(rt.stop, 5),
            "alvo": round(rt.target, 5),
            "unidades": round(rt.units, 2),
            "risco_$": round(rt.risk_cash, 2),
            "ganho_potencial_$": round(rt.reward_cash, 2),
            "cobre_meta": f"{rt.target_coverage:.0%}",
            "R/R": round(rt.rr, 2) if np.isfinite(rt.rr) else np.nan,
        } for rt in routes])
        st.dataframe(table, hide_index=True, use_container_width=True)
        for rt in routes:
            with st.expander(f"Roteiro {rt.priority}: {rt.action} {rt.symbol} · {rt.status}", expanded=rt.priority == 1):
                for step in rt.roteiro:
                    st.markdown(f"- {step}")
                c = st.columns(3)
                if c[0].button(f"Registrar intenção {rt.symbol}", key=f"directed_intent_{rt.symbol}_{rt.priority}"):
                    ts = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
                    row = {
                        "id": signal_id(rt.symbol, ts, rt.action, rt.entry),
                        "timestamp": ts,
                        "mode": "DIRIGIDO",
                        "symbol": rt.symbol,
                        "operator_choice": "ROTEIRO GERADO",
                        "actual_action": rt.action,
                        "real_price": rt.entry,
                        "user_confidence": 50,
                        "operator_note": "Intenção registrada a partir do modo dirigido.",
                        "target_profit_goal": goal.target_profit_cash,
                        "capital_goal": goal.capital_cash,
                        "max_loss_goal": goal.max_loss_cash,
                        "route_score": rt.route_score,
                        "route_units": rt.units,
                        "route_risk_cash": rt.risk_cash,
                        "route_reward_cash": rt.reward_cash,
                    }
                    append_decision_log(row)
                    st.success("Intenção registrada no log de decisões.")

# ─────────────────────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────────────────────

def safe_clear_cache(*funcs) -> None:
    for f in funcs:
        clear = getattr(f, "clear", None)
        if callable(clear): clear()

def explain_one_line(r: CoachReport, decision: str, reason: str) -> str:
    if index_market_closed_or_paused_for(r.symbol, r.session):
        return f"Mercado/futuros fechados ou em pausa. O preço e a tendência vêm do último candle disponível; use apenas para revisão e preparação, não para entrada ou saída operacional. {reason}"
    if decision == "COMPRAR":
        return f"Compra só se confirmar: tendência {r.trend.lower()} {r.trend_strength.lower()}, score comprador {r.score_bull}×{r.score_bear}; {reason.lower()}"
    if decision == "VENDER":
        return f"Venda só se confirmar: tendência {r.trend.lower()} {r.trend_strength.lower()}, score vendedor {r.score_bear}×{r.score_bull}; {reason.lower()}"
    if decision == "SAIR · PROTEGER":
        return f"Prioridade é defesa: risco de reversão {r.reversal_risk.lower()} e sessão {r.session.name.lower()}."
    return f"Aguardar: confluência insuficiente ou bloqueada. {reason}"

def render_context(ctx: ContextReport) -> None:
    with st.container(border=True):
        cols = st.columns(3)
        cols[0].metric("Regime macro", ctx.risk_regime, f"score {ctx.risk_score:+.2f}")
        cols[1].metric("Notícias", ctx.news_pressure, f"score {ctx.news_score:+.2f}")
        cols[2].metric("Leitura", "prudente" if ctx.risk_regime == "DEFENSIVO" else "normal")
        notes = ctx.macro_notes + ctx.cross_asset_notes
        if notes:
            st.caption(" · ".join(notes[:4]))

def render_simple_card(r: CoachReport, ml: MLReport, decision: str, score: float, reason: str, plan: RiskPlan, strategy: TimeframeStrategy) -> None:
    with st.container(border=True):
        c = st.columns([0.8, 1.0, 1.0, 1.0, 1.0])
        c[0].subheader(r.symbol)
        c[1].metric("Agora", action_badge(decision), f"score {score:.2f}")
        c[2].metric("Preço", fmt_price(r.price, r.asset_type), f"{r.change_pct:+.2f}%")
        c[3].metric("Tendência", f"{r.trend} {r.trend_strength}", f"rev. {r.reversal_risk}")
        c[4].metric("ML", ml.prediction if ml.available else "sem sinal", f"p={ml.calibrated_probability:.0%}" if ml.available and np.isfinite(ml.calibrated_probability) else "")
        if index_market_closed_or_paused_for(r.symbol, r.session):
            st.info("Mercado/futuros fechados ou em pausa: os indicadores abaixo são do último candle disponível. O app bloqueia coaching operacional e alertas ao vivo para índices nessa condição.")
        st.write(explain_one_line(r, decision, reason))
        if ml.conformal_available and len(ml.conformal_set) > 1:
            st.caption(f"⚠ Incerteza validada: o modelo não descarta {' / '.join(ml.conformal_set)} com confiança neste momento (ver detalhes → ML).")
        tfc = st.columns([1.2, 1.0, 1.0, 1.0])
        tfc[0].metric("Janela", strategy.label)
        tfc[1].metric("Estilo", strategy.style)
        tfc[2].metric("Dados/ML", f"{strategy.data_interval} · {strategy.horizon_bars} barras")
        tfc[3].metric("Risco relativo", f"{strategy.risk_factor:.0%}")
        with st.expander("Estratégia apropriada para esta janela", expanded=False):
            st.markdown(timeframe_strategy_markdown(strategy))
        if decision in {"COMPRAR", "VENDER"}:
            p = st.columns(4)
            p[0].metric("Stop", fmt_price(plan.stop, r.asset_type))
            p[1].metric("Alvo", fmt_price(plan.target, r.asset_type))
            p[2].metric("R/R", f"{plan.rr:.2f}" if np.isfinite(plan.rr) else "—")
            p[3].metric("Unid. por risco", f"{plan.suggested_units:.2f}" if np.isfinite(plan.suggested_units) else "—")
            st.caption("Invalidação: " + plan.invalidation)
        else:
            st.caption("Sem plano de entrada: preservar atenção e aguardar nova confluência.")

def render_chart(df: pd.DataFrame, r: CoachReport) -> None:
    d = add_indicators(df).dropna().tail(180)
    if d.empty: return
    if go is not None:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Preço"))
        fig.add_trace(go.Scatter(x=d.index, y=d["ema21"], name="EMA21", mode="lines"))
        fig.add_trace(go.Scatter(x=d.index, y=d["ema50"], name="EMA50", mode="lines"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{r.symbol}")
    else:
        st.line_chart(d[["close", "ema21", "ema50"]], height=320)

def render_backtest_panel(symbols: list[str], period: str, interval: str, horizon: int, model_kind: str, strategy: TimeframeStrategy) -> None:
    with st.expander("Laboratório: walk-forward operacional, gates e contextos", expanded=False):
        st.caption("Treina no passado, prevê bloco seguinte e simula stop/alvo em ATR com custo e slippage.")
        st.info(f"Janela ativa: {strategy.label} · dados {interval} · horizonte {horizon} barras · stop sugerido {strategy.stop_atr:.2f} ATR · alvo sugerido {strategy.target_atr:.2f} ATR")
        c = st.columns(7)
        bt_symbol = c[0].selectbox("Ativo", symbols or DEFAULT_SYMBOLS)
        min_train = c[1].slider("Treino", 80, 600, 200, step=20)
        step = c[2].slider("Passo", 1, 12, max(3, horizon))
        cost = c[3].slider("Custo bps", 0.0, 25.0, 2.0, step=0.5)
        slip = c[4].slider("Slippage bps", 0.0, 20.0, 1.0, step=0.5)
        stop = c[5].slider("Stop ATR", 0.25, 3.0, 0.75, step=0.05)
        target = c[6].slider("Alvo ATR", 0.25, 5.0, 1.25, step=0.05)
        if st.button("Rodar laboratório", key="run_lab"):
            bt, mt = walk_forward_backtest(bt_symbol, period, interval, horizon, model_kind, min_train, step, cost, stop, target, slip)
            if bt.empty:
                st.warning(mt.get("erro", "Sem resultado."))
                return
            st.json(mt)
            st.line_chart(bt["strategy_ret_pct"].cumsum(), height=220)
            tabs = st.tabs(["Sessão", "Regime", "Direção", "Volatilidade", "Amostra"])
            for tab, ctx in zip(tabs[:4], ["session", "technical_regime", "prediction", "vol_regime"]):
                with tab:
                    st.dataframe(_context_summary(bt, ctx), hide_index=True, use_container_width=True)
            with tabs[4]:
                st.dataframe(bt.tail(80), hide_index=True, use_container_width=True)
                st.download_button("Baixar backtest CSV", bt.to_csv(index=False).encode("utf-8"), file_name=f"backtest_{bt_symbol}.csv", mime="text/csv")

def render_details(r: CoachReport, ml: MLReport, opinions: list[AgentOpinion], weights: dict[str, float], audit: DecisionAudit, df: pd.DataFrame) -> None:
    with st.expander(f"Detalhes técnicos · {r.symbol}", expanded=False):
        tabs = st.tabs(["Gráfico", "Agentes", "ML", "Auditoria", "Dados"])
        with tabs[0]: render_chart(df, r)
        with tabs[1]: st.dataframe(pd.DataFrame([asdict(op) for op in opinions]), hide_index=True, use_container_width=True)
        with tabs[2]:
            st.write(ml.explanation)
            if ml.model_table: st.dataframe(pd.DataFrame(ml.model_table), hide_index=True, use_container_width=True)
            if ml.top_features: st.dataframe(pd.DataFrame(ml.top_features), hide_index=True, use_container_width=True)
            st.json({"class_balance": ml.class_balance, "quality": ml.model_quality, "f1": ml.f1_macro, "mcc": ml.mcc})
            st.divider()
            st.markdown("**Incerteza (conformal prediction)**")
            if ml.conformal_available:
                set_txt = " ∪ ".join(ml.conformal_set) if ml.conformal_set else "—"
                st.markdown(f"Conjunto de previsão (cobertura alvo {1-ml.conformal_alpha:.0%}): **{set_txt}**")
                if len(ml.conformal_set) == 1:
                    st.caption("Conjunto com 1 classe: o modelo tem informação suficiente, validada empiricamente, para descartar as alternativas neste ponto.")
                elif len(ml.conformal_set) >= len(ml.class_balance):
                    st.caption("Conjunto inclui todas as classes: não há informação suficiente, validada empiricamente, para descartar nenhuma alternativa agora — trate a previsão pontual com cautela extra.")
                else:
                    st.caption("Conjunto parcial: o modelo consegue descartar pelo menos uma classe com confiança validada, mas ainda há ambiguidade real.")
                st.caption(ml.conformal_note)
            else:
                st.caption(f"Conformal prediction indisponível: {ml.conformal_note or 'sem motivo informado'}.")
        with tabs[3]:
            st.dataframe(pd.DataFrame([weights]), hide_index=True, use_container_width=True)
            st.json(asdict(audit))
        with tabs[4]: st.json(asdict(r.data_health))

def main() -> None:
    st.set_page_config(page_title="F-Coach V8.4 Fechado Cache Safe", layout="wide")
    st.title(f"F-Coach Full V{APP_VERSION}")
    st.caption("Motor elaborado, tela simples: decisão, motivo, risco e contexto. Não executa ordens.")

    with st.sidebar:
        st.header("Painel simples")
        tradable_symbols = [k for k in TICKER_MAP if k not in CONTEXT_SYMBOLS]
        symbols = st.multiselect("Ativos", tradable_symbols, default=DEFAULT_STARTUP_SYMBOLS)
        with st.expander("Núcleo padrão", expanded=False):
            st.markdown("**Índices US:** US30, US500, US100")
            st.markdown("**Pares FX estáveis incluídos:** EURUSD, USDCHF, EURCHF")
            st.caption("Esses pares entram no default por terem liquidez/estabilidade relativa mais favoráveis para aumentar acerto bruto. O app ainda exige dados OK, risco controlado e confluência mínima.")
        trade_minutes = st.selectbox("Janela da operação", TIMEFRAME_WINDOWS, index=4, format_func=timeframe_label)
        strategy = timeframe_strategy(int(trade_minutes))
        auto_timeframe = st.toggle("Calibrar automaticamente pela janela", value=True)
        if auto_timeframe:
            period = strategy.default_period
            interval = strategy.data_interval
            horizon = strategy.horizon_bars
            st.caption(f"Auto: histórico {period}, candles {interval}, horizonte ML {horizon} barras.")
        else:
            period = st.selectbox("Janela histórica", ["7d", "30d", "60d", "90d", "6mo", "1y"], index=3)
            interval = st.selectbox("Intervalo", ["1m", "5m", "15m", "30m", "1h", "1d"], index=4)
            horizon = st.slider("Horizonte em barras", 1, 20, strategy.horizon_bars)
        with st.expander("Estratégia da janela selecionada", expanded=True):
            st.json(timeframe_overview(strategy))
            st.caption(timeframe_strategy_markdown(strategy))
        live = st.toggle("Atualização automática", value=False, help="Desligado por padrão para evitar reruns pesados na abertura.")
        refresh_seconds = st.slider("Atualizar a cada", 60, 900, 300, step=60)
        st.header("Motor")
        model_kind = st.selectbox("ML", ["Auto Ensemble", "Random Forest", "Extra Trees", "Gradient Boosting", "Logístico"], index=4, help="Logístico abre mais rápido. Use Auto Ensemble quando quiser maior robustez, mas ele é pesado.")
        run_ml = st.toggle("Rodar ML/Conformal agora", value=False, help="Desligado por padrão: a abertura fica rápida. Ligue para treinar ML, conformal e Bayes com base no ML.")
        max_ml_assets = st.slider("Máximo de ativos com ML por rodada", 1, 6, 1, help="Evita treinar vários modelos em muitos ativos ao mesmo tempo.")
        strictness = st.slider("Rigor", 0.0, 1.0, 0.65, step=0.05)
        bayes_prior_profile = st.selectbox(
            "Prior Bayes contextual",
            ["Conservador", "Neutro", "Agressivo"],
            index=0,
            help="Controla apenas o prior inicial. Assim que houver trades reais fechados por ativo×janela×sessão×direção, a evidência domina."
        )
        use_gates = st.toggle("Usar gates aprendidos", value=True, help="Não recalcula na abertura. Usa a última calibração salva, se existir; caso contrário, fica inativo até você calibrar.")
        gate_scope = st.radio("Calibração dos gates", ["Leve: primeiro ativo selecionado", "Completa: todos os ativos selecionados"], index=0, horizontal=False, help="Use o modo leve para abrir e testar rápido; rode completa quando já estiver satisfeito com ativos e timeframe.")
        calibrate_gates_now = st.button("Calibrar gates agora", help="Roda o walk-forward contextual e salva o resultado em JSON. Pode demorar, sobretudo com muitos ativos e candles curtos.")
        delete_gates_now = st.button("Apagar gates salvos", help="Remove a última calibração persistida. Não apaga diário nem trades.")
        include_news = st.toggle("Macro/notícias", value=False, help="Desligado por padrão para abertura rápida. Ligue quando quiser incluir VIX/DXY/yields/petróleo e notícias.")
        st.header("Plano de risco")
        risk_cash_base = st.number_input("Risco financeiro base por trade", min_value=1.0, value=100.0, step=10.0)
        risk_cash = float(risk_cash_base) * strategy.risk_factor
        st.caption(f"Risco efetivo pela janela: {risk_cash:.2f} ({strategy.risk_factor:.0%} do risco base).")
        if auto_timeframe:
            stop_atr = strategy.stop_atr
            target_atr = strategy.target_atr
            st.caption(f"Stop/alvo automáticos: {stop_atr:.2f} ATR / {target_atr:.2f} ATR.")
        else:
            stop_atr = st.slider("Stop padrão (ATR)", 0.25, 3.0, strategy.stop_atr, step=0.05)
            target_atr = st.slider("Alvo padrão (ATR)", 0.25, 5.0, strategy.target_atr, step=0.05)

        st.header("Modo dirigido")
        directed_enabled = st.toggle("Ativar modo dirigido por meta", value=False, help="Gera roteiros condicionais com base no capital alocado, na meta de lucro e na perda máxima planejada.")
        directed_capital = st.number_input("Capital que pretendo alocar", min_value=1.0, value=1000.0, step=100.0)
        directed_target = st.number_input("Lucro pretendido", min_value=1.0, value=100.0, step=10.0)
        directed_max_loss = st.number_input("Perda máxima antes de parar", min_value=1.0, value=max(20.0, float(directed_capital) * 0.02), step=10.0)
        directed_max_trades = st.slider("Máximo de operações no roteiro", 1, 5, 2)
        directed_point_value = st.number_input("Valor financeiro por ponto/unidade", min_value=0.01, value=1.0, step=0.5)
        directed_max_units = st.number_input("Máximo de unidades/contratos", min_value=0.01, value=10.0, step=1.0)
        directed_min_rr = st.slider("R/R mínimo para roteiro", 0.5, 3.0, 1.30, step=0.05)
        directed_partial = st.toggle("Permitir roteiros parciais da meta", value=True)
        directed_goal = DirectedGoal(
            enabled=bool(directed_enabled),
            capital_cash=float(directed_capital),
            target_profit_cash=float(directed_target),
            max_loss_cash=float(directed_max_loss),
            max_trades=int(directed_max_trades),
            point_value=float(directed_point_value),
            min_rr=float(directed_min_rr),
            max_units=float(directed_max_units),
            allow_partial_targets=bool(directed_partial),
        )
        if directed_enabled:
            st.caption(summarize_directed_goal(directed_goal))

        st.header("Copiloto IA")
        ai_provider = st.selectbox("Provedor", ["Heurístico local", "OpenAI compatível", "Groq", "Ollama local"], index=0)
        ai_model = st.text_input("Modelo", value={"Heurístico local": "", "OpenAI compatível": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "Groq": "llama-3.3-70b-versatile", "Ollama local": "llama3.1"}.get(ai_provider, ""))
        ai_base_url = st.text_input("Base URL", value="http://localhost:11434" if ai_provider == "Ollama local" else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") if ai_provider == "OpenAI compatível" else "")
        ai_api_key = st.text_input("API key opcional", value="", type="password", help="Também aceita OPENAI_API_KEY ou GROQ_API_KEY no ambiente.")
        st.header("Alertas")
        alerts_enabled = st.toggle("Telegram em alta convicção", value=False)
        if alerts_enabled and not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
            st.warning("Configure F_COACH_TELEGRAM_TOKEN e F_COACH_TELEGRAM_CHAT_ID.")
        if st.button("Testar Telegram agora"):
            ok, status = send_telegram_alert(
                "Teste do F-Coach: Telegram configurado e funcionando."
            )
            if ok:
                st.success(f"Telegram OK: {status}")
            else:
                st.error(f"Telegram falhou: {status}")


        if st.button("Atualizar agora"):
            safe_clear_cache(fetch, fetch_market_news, walk_forward_backtest, learned_context_gates)
            st.rerun()

    sess = market_session()
    st.info(f"Sessão: **{sess.name}** · {sess.note} · Brasília {datetime.now(LOCAL_TZ).strftime('%d/%m/%Y %H:%M:%S')} · NY {datetime.now(ET).strftime('%H:%M')}")
    if include_news:
        ctx = build_context_report(include_news)
    else:
        ctx = ContextReport("NEUTRO", 0.0, ["Macro/notícias desligado para abertura rápida."], [], "NEUTRO", 0.0, [])
    render_context(ctx)

    # Gates contextuais persistentes: não rodam automaticamente na abertura.
    # A V8.1 carrega o último JSON salvo e só recalibra quando o botão é acionado.
    if "learned_gates_v81" not in st.session_state:
        st.session_state["learned_gates_v81"] = load_gates_cache()

    current_gate_signature = gates_signature(symbols, period, interval, horizon, model_kind, stop_atr, target_atr, strictness)

    if delete_gates_now:
        ok, msg = delete_gates_cache()
        st.session_state["learned_gates_v81"] = {"available": False, "reason": "gates apagados" if ok else msg}
        st.toast(msg if ok else f"Falha ao apagar gates: {msg}")

    gates = {"available": False, "reason": "desligado"}
    if use_gates and symbols:
        if calibrate_gates_now:
            if gate_scope.startswith("Leve"):
                calibration_symbols = tuple(symbols[:1])
            else:
                calibration_symbols = tuple(symbols)
            calibration_signature = gates_signature(calibration_symbols, period, interval, horizon, model_kind, stop_atr, target_atr, strictness)
            with st.spinner(f"Calibrando gates contextuais para {', '.join(calibration_symbols)}... use o modo leve para rapidez."):
                calibrated = learned_context_gates(
                    calibration_symbols, period, interval, horizon, model_kind,
                    200, max(3, horizon), 2.0, stop_atr, target_atr, 1.0, strictness,
                )
                ok, msg = save_gates_cache(calibrated, calibration_signature)
                if ok:
                    calibrated["_cache_meta"] = {"created_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"), "signature": calibration_signature, "path": msg}
                    st.session_state["learned_gates_v81"] = calibrated
                    st.success(f"Gates calibrados e salvos em {msg}")
                else:
                    st.session_state["learned_gates_v81"] = calibrated
                    st.warning(f"Gates calibrados, mas não foi possível salvar: {msg}")

        gates = st.session_state.get("learned_gates_v81", {"available": False, "reason": "ainda não calibrado"})
        saved_sig = gates.get("_cache_meta", {}).get("signature", {}) if isinstance(gates, dict) else {}
        same_config = signature_match(saved_sig, current_gate_signature) if saved_sig else False
        if gates.get("available"):
            if not same_config:
                st.warning("Há gates salvos, mas eles foram calibrados com configuração diferente da atual. Eles serão usados com cautela; recalibre quando possível.")
            with st.expander("Gates aprendidos ativos", expanded=False):
                st.caption(gates_age_label(gates))
                st.json({
                    "arquivo": gates.get("_cache_meta", {}).get("path", str(GATES_CACHE_PATH)),
                    "configuração_compatível": bool(same_config),
                    "contextos_ruins": gates.get("poor", {}),
                    "contextos_bons": gates.get("good", {}),
                    "assinatura_salva": saved_sig,
                })
                metrics_df = pd.DataFrame(gates.get("metrics", []))
                if not metrics_df.empty:
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        else:
            st.caption("Gates contextuais ligados, mas sem calibração ativa: " + gates.get("reason", "sem resultado"))

    render_backtest_panel(symbols, period, interval, horizon, model_kind, strategy)

    run_every = refresh_seconds if live else None

    @st.fragment(run_every=run_every)
    def live_panel() -> None:
        if not symbols:
            st.warning("Selecione pelo menos um ativo.")
            return
        reports: list[tuple[CoachReport, MLReport, pd.DataFrame]] = []
        for i_sym, sym in enumerate(symbols):
            df = fetch(sym, period, interval)
            if df.empty:
                st.warning(f"Sem dados para {sym}.")
                continue
            try:
                r = analyze(df, sym, interval)
                if run_ml and i_sym < max_ml_assets:
                    with st.spinner(f"Treinando ML/Conformal para {sym}..."):
                        ml = train_predict_ml(sym, period, interval, horizon, model_kind)
                else:
                    ml = MLReport(sym, False, explanation="ML/Conformal desligado nesta rodada para abertura rápida.")
                reports.append((r, ml, df))
            except Exception as exc:
                st.error(f"Erro em {sym}: {exc}")
        if not reports:
            return
        only_reports = [r for r, _, _ in reports]
        decisions = {}
        rows = []
        directed_packs: list[tuple[CoachReport, MLReport, str, float, str, RiskPlan, DecisionAudit]] = []
        current_prices = {}
        for r, ml, df in reports:
            opinions = [technical_agent(r), risk_agent(r), session_agent(r), timeframe_agent(strategy, r), ml_agent(ml), conformal_uncertainty_agent(ml), bayesian_context_agent(r, ml, strategy, bayes_prior_profile), relative_strength_agent(only_reports, r), forex_stability_agent(r), macro_agent(ctx, r)]
            decision, score, reason, weights, audit = synthesize_agents(opinions, r, ml, ctx, gates, strictness)
            plan = build_risk_plan(r, decision, risk_cash, stop_atr, target_atr)
            decisions[r.symbol] = (decision, score, reason, weights, audit, opinions, plan)
            directed_packs.append((r, ml, decision, score, reason, plan, audit))
            current_prices[r.symbol] = r.price
            rows.append({"ativo": r.symbol, "classe": forex_stability_label(r.symbol) if r.asset_type == "forex" else r.asset_type, "decisão": decision, "score": round(score, 2), "preço": fmt_price(r.price, r.asset_type), "var_%": round(r.change_pct, 2), "tendência": f"{r.trend} {r.trend_strength}", "risco": r.reversal_risk, "ML": ml.prediction, "conformal": "∪".join(ml.conformal_set) if ml.conformal_available else "—", "qual_ML": round(ml.model_quality, 2) if np.isfinite(ml.model_quality) else np.nan, "dados": r.data_health.status, "janela": strategy.label})
        st.subheader("Resumo executivo")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        render_directed_mode_panel(directed_packs, directed_goal, strategy)
        st.subheader("Agora")
        for r, ml, df in reports:
            decision, score, reason, weights, audit, opinions, plan = decisions[r.symbol]
            render_simple_card(r, ml, decision, score, reason, plan, strategy)
            render_copilot_and_realtime_log(r, ml, decision, score, reason, plan, audit, ai_provider, ai_model, ai_api_key, ai_base_url, strategy)
            if alerts_enabled:
                status = maybe_send_high_conviction_alert(r, ml, decision, score, plan, strategy, directed_goal, audit)
                if status: st.toast(f"{r.symbol}: {status}")
        st.subheader("Detalhes sob demanda")
        for r, ml, df in reports:
            decision, score, reason, weights, audit, opinions, plan = decisions[r.symbol]
            render_details(r, ml, opinions, weights, audit, df)
            with st.expander(f"Registrar no diário · {r.symbol}", expanded=False):
                note = st.text_input("Nota", key=f"note_{r.symbol}")
                if st.button(f"Salvar sinal {r.symbol}", key=f"save_{r.symbol}"):
                    ts = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
                    row = {"id": signal_id(r.symbol, ts, decision, r.price), "timestamp": ts, "symbol": r.symbol, "price": r.price, "decision": decision, "confidence": score, "trend": f"{r.trend} {r.trend_strength}", "reversal_risk": r.reversal_risk, "ml_prediction": ml.prediction, "ml_quality": ml.model_quality, "macro_regime": ctx.risk_regime, "news_pressure": ctx.news_pressure, "stop": plan.stop, "target": plan.target, "rr": plan.rr, "note": note}
                    st.success("Salvo no diário." if append_journal(row) else "Já estava salvo.")
        st.subheader("Diário operacional")
        journal = evaluate_journal(current_prices)
        if journal.empty:
            st.caption("Nenhum sinal registrado ainda.")
        else:
            st.dataframe(journal.tail(40), hide_index=True, use_container_width=True)
            st.download_button("Baixar diário CSV", journal.to_csv(index=False).encode("utf-8"), "f_coach_journal_v7.csv", "text/csv")
        with st.expander("Decisões aceitas/recusadas e P&L real", expanded=False):
            dlog = load_decision_log()
            tlog = load_trade_log()
            tabs = st.tabs(["Decisões", "Trades/P&L", "Resumo P&L"])
            with tabs[0]:
                if dlog.empty:
                    st.caption("Nenhuma decisão registrada ainda.")
                else:
                    st.dataframe(dlog.tail(80), hide_index=True, use_container_width=True)
                    st.download_button("Baixar decisões CSV", dlog.to_csv(index=False).encode("utf-8"), "f_coach_decisions_v7.csv", "text/csv")
            with tabs[1]:
                if tlog.empty:
                    st.caption("Nenhum trade registrado ainda.")
                else:
                    st.dataframe(tlog.tail(80), hide_index=True, use_container_width=True)
                    st.download_button("Baixar trades CSV", tlog.to_csv(index=False).encode("utf-8"), "f_coach_trades_v7.csv", "text/csv")
            with tabs[2]:
                if tlog.empty or "realized_pnl" not in tlog.columns:
                    st.caption("Resumo será exibido após fechamento de posições.")
                else:
                    closed = tlog[tlog.get("status", pd.Series(dtype=str)).astype(str) == "FECHADA"].copy()
                    if closed.empty:
                        st.caption("Ainda não há posições fechadas.")
                    else:
                        closed["realized_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce")
                        summary = closed.groupby(["symbol", "side"], dropna=False).agg(
                            trades=("id", "count"),
                            pnl_total=("realized_pnl", "sum"),
                            pnl_medio=("realized_pnl", "mean"),
                            win_rate=("realized_pnl", lambda x: float((x > 0).mean())),
                        ).reset_index()
                        st.dataframe(summary, hide_index=True, use_container_width=True)

    live_panel()

    st.divider()
    st.markdown("""
**Leitura correta.** A V8.3 abre em modo leve: ML/Conformal e macro ficam sob demanda. Quando ativados, o sistema continua deliberadamente conservador e sensível à janela temporal: prefere `AGUARDAR` quando os agentes discordam, quando os dados estão atrasados, quando a sessão é ruim ou quando os gates históricos indicam contexto fraco. O objetivo é reduzir falsos sinais, não aumentar a quantidade de entradas.
""")

if __name__ == "__main__":
    main()
