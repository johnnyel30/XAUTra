"""
╔══════════════════════════════════════════════════════════════════════╗
║   XAUUSDT ORDER BLOCK BOT  ·  v1.0                                   ║
║   Estrategia : BOS + Order Block  ·  Marco: M15                      ║
║   Exchange   : Binance Futures USDT-M  ·  One-Way Mode               ║
║   Riesgo     : 1% fijo  ·  R:R 1:2.5  ·  Break-even automático      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
#  DEPENDENCIAS
#  pip install ccxt pandas numpy python-dotenv requests
# ══════════════════════════════════════════════════════════════════════
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import os
import logging
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum, auto
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 · CONFIGURACIÓN CENTRAL
# ══════════════════════════════════════════════════════════════════════
load_dotenv()

# ── Credenciales (cargadas desde .env) ────────────────────────────────
API_KEY    = os.getenv("API_KEY",    "")
API_SECRET = os.getenv("API_SECRET", "")
TG_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TG_CHAT    = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Instrumento ───────────────────────────────────────────────────────
# Binance Futures USDT-M usa el formato "XAU/USDT:USDT" en ccxt
CCXT_SYMBOL = "XAU/USDT:USDT"
TIMEFRAME   = "15m"
LEVERAGE    = 10            # Ajustado dinámicamente por el motor de riesgo

# ── Parámetros de Estrategia ──────────────────────────────────────────
PIVOT_K        = 5          # Ventana de pivotes (velas a cada lado)
ATR_LEN        = 14         # Período ATR principal
EMA_LEN        = 55         # EMA de tendencia
ADX_LEN        = 14         # ADX para filtro de tendencia
IMPULSE_MULT   = 1.5        # Cuerpo de vela >= IMPULSE_MULT × ATR para BOS válido
OB_ENTRY_PCT   = 0.50       # Entrada al 50% del rango del OB (Consequent Encroachment)
MAX_OB_RETESTS = 1          # Máximo 1 retest; el 2do contacto invalida el OB

# ── Gestión de Riesgo ─────────────────────────────────────────────────
RISK_PCT        = 0.01      # Riesgo máximo por operación: 1% del equity
SL_ATR_PAD      = 0.20      # Colchón SL = 20% del ATR (evita stop hunts)
MIN_RR          = 2.5       # R:R mínimo; si no cabe, se descarta el setup
DAILY_LOSS_LIM  = 0.03      # Detener el día al -3% del equity de apertura
WEEKLY_LOSS_LIM = 0.06      # Detener la semana al -6% del equity de apertura

# ── Filtros de Protección ─────────────────────────────────────────────
ADX_MIN        = 25         # ADX < 25 = mercado lateral = no operar
ADX_LATERAL_N  = 20         # Ventana de velas para detectar rango estrecho
VOL_FAST_LEN   = 5          # ATR rápido (filtro de volatilidad extrema)
VOL_SLOW_LEN   = 20         # ATR lento
VOL_MAX_RATIO  = 2.50       # Si ATR_fast/ATR_slow > 2.5 = volatilidad caótica
EMA_SLOPE_N    = 5          # Barras para calcular la pendiente de la EMA55
NEWS_PAUSE_MIN = 15         # Minutos de pausa antes y después de noticias alto impacto
SPREAD_HIST_N  = 100        # Historial de spreads para calcular percentil 95

# ── Sesión Horaria (GMT) ──────────────────────────────────────────────
SESSION_START  = 13         # 13:00 GMT — solapamiento Londres / Nueva York
SESSION_END    = 17         # 17:00 GMT

# ── Infraestructura ───────────────────────────────────────────────────
CANDLE_LIMIT   = 220        # Velas a descargar (más que suficiente para indicadores)
LOOP_SEC       = 15         # Segundos entre ciclos del loop principal
MAX_RETRIES    = 5          # Reintentos máximos para llamadas API
RETRY_BASE     = 2.0        # Base del exponential backoff (segundos)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_xauusdt.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("OBBot")


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 · ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════

class BotState(Enum):
    IDLE    = auto()   # Sin posición, buscando setup
    PENDING = auto()   # Orden límite enviada, esperando fill
    ACTIVE  = auto()   # Posición abierta — gestionando SL/BE
    STOPPED = auto()   # Límite de pérdida alcanzado — hibernación


@dataclass
class OrderBlock:
    """Zona de liquidez institucional identificada algorítmicamente."""
    direction:  str    # "bull" | "bear"
    ob_high:    float  # Máximo de la vela OB
    ob_low:     float  # Mínimo de la vela OB
    ob_mid:     float  # 50% del rango — precio de entrada (CE)
    atr_at_bos: float  # ATR en el momento del BOS — usado para SL/TP
    retests:    int = 0
    valid:      bool = True


@dataclass
class Trade:
    """Parámetros completos de una operación activa."""
    direction:    str
    entry_price:  float
    sl_price:     float
    tp_price:     float
    qty:          float
    risk_usd:     float
    entry_order_id: Optional[str] = None
    sl_order_id:    Optional[str] = None
    tp_order_id:    Optional[str] = None
    be_triggered:   bool = False     # True una vez que se mueve el SL a break-even


@dataclass
class RiskTracker:
    """Rastrea el PnL diario y semanal para los límites de pérdida."""
    equity_day:    float = 0.0
    equity_week:   float = 0.0
    pnl_day:       float = 0.0
    pnl_week:      float = 0.0
    reset_day:     str   = ""
    reset_week:    str   = ""


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 · TELEGRAM
# ══════════════════════════════════════════════════════════════════════

def tg(msg: str):
    """Envía notificación a Telegram. Falla silenciosamente si no está configurado."""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram: {e}")


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 · INDICADORES TÉCNICOS
# ══════════════════════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c).abs(), (df["low"] - c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """ADX con suavizado de Wilder."""
    h, l, ph, pl = df["high"], df["low"], df["high"].shift(1), df["low"].shift(1)
    pc = df["close"].shift(1)

    up   = h - ph
    down = pl - l
    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)

    tr   = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=period, adjust=False).mean()

    di_p = 100 * pd.Series(dm_p, index=df.index).ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_m = 100 * pd.Series(dm_m, index=df.index).ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx   = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)).fillna(0)
    return dx.ewm(span=period, adjust=False).mean()


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Añade todos los indicadores al dataframe de velas."""
    df = df.copy()
    df["atr"]       = calc_atr(df, ATR_LEN)
    df["atr_fast"]  = calc_atr(df, VOL_FAST_LEN)
    df["atr_slow"]  = calc_atr(df, VOL_SLOW_LEN)
    df["ema55"]     = calc_ema(df["close"], EMA_LEN)
    df["adx"]       = calc_adx(df, ADX_LEN)
    df["ema_slope"] = df["ema55"] - df["ema55"].shift(EMA_SLOPE_N)
    df["body"]      = (df["close"] - df["open"]).abs()
    return df.dropna()


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 5 · DETECCIÓN DE PIVOTES Y ORDER BLOCKS
# ══════════════════════════════════════════════════════════════════════

def pivot_highs(df: pd.DataFrame, k: int = PIVOT_K) -> List[int]:
    """Índices de pivotes altos confirmados con k velas a cada lado."""
    h = df["high"].values
    return [i for i in range(k, len(h) - k) if h[i] == max(h[i - k : i + k + 1])]


def pivot_lows(df: pd.DataFrame, k: int = PIVOT_K) -> List[int]:
    """Índices de pivotes bajos confirmados con k velas a cada lado."""
    l = df["low"].values
    return [i for i in range(k, len(l) - k) if l[i] == min(l[i - k : i + k + 1])]


def find_order_blocks(df: pd.DataFrame) -> List[OrderBlock]:
    """
    Detecta Order Blocks válidos en el dataframe enriquecido.

    Algoritmo:
    1. Identifica pivotes estructurales (PH y PL).
    2. Busca la primera vela posterior que rompe el pivote con cierre (BOS).
    3. Valida el BOS: cuerpo >= IMPULSE_MULT×ATR, ADX>25, EMA55 alineada.
    4. Retrocede para aislar la vela con el mínimo/máximo extremo (el OB).
    5. Descarta OBs ya mitigados (precio cerró a través de ellos).
    """
    obs: List[OrderBlock] = []
    ph_idx_list = pivot_highs(df)
    pl_idx_list = pivot_lows(df)

    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values
    adxs   = df["adx"].values
    ema55s = df["ema55"].values
    slopes = df["ema_slope"].values
    af     = df["atr_fast"].values
    as_    = df["atr_slow"].values
    n      = len(df)

    # ── Bullish Order Blocks ──────────────────────────────────────────
    for ph in ph_idx_list:
        ph_price = highs[ph]
        for bos in range(ph + 1, n):
            if closes[bos] <= ph_price:
                continue                           # Sin BOS todavía

            # ── Validaciones del impulso ─────────────────────────
            body = closes[bos] - opens[bos]
            if body < IMPULSE_MULT * atrs[bos]:
                break                              # Impulso débil → descartar PH

            if adxs[bos] < ADX_MIN:
                break                              # Mercado lateral en el BOS

            if closes[bos] <= ema55s[bos] or slopes[bos] <= 0:
                break                              # Precio o EMA no alcista

            if as_[bos] > 0 and (af[bos] / as_[bos]) > VOL_MAX_RATIO:
                break                              # Volatilidad caótica

            # ── Aislar el OB: vela con el mínimo más bajo ─────────
            # Buscamos el pivot low previo al PH como inicio del segmento
            start = next((pl for pl in reversed(pl_idx_list) if pl < ph), max(0, ph - 30))
            segment = range(start, bos)

            ob_idx = min(segment, key=lambda i: lows[i], default=None)
            if ob_idx is None:
                break

            ob_h = highs[ob_idx]
            ob_l = lows[ob_idx]

            # ── Verificar que no está mitigado ────────────────────
            if any(closes[i] < ob_l for i in range(bos, n)):
                break   # Ya fue mitigado; no colocar orden aquí

            obs.append(OrderBlock(
                direction  = "bull",
                ob_high    = ob_h,
                ob_low     = ob_l,
                ob_mid     = ob_l + (ob_h - ob_l) * OB_ENTRY_PCT,
                atr_at_bos = atrs[bos],
            ))
            break  # Un BOS por pivote es suficiente

    # ── Bearish Order Blocks ──────────────────────────────────────────
    for pl in pl_idx_list:
        pl_price = lows[pl]
        for bos in range(pl + 1, n):
            if closes[bos] >= pl_price:
                continue

            body = opens[bos] - closes[bos]
            if body < IMPULSE_MULT * atrs[bos]:
                break

            if adxs[bos] < ADX_MIN:
                break

            if closes[bos] >= ema55s[bos] or slopes[bos] >= 0:
                break

            if as_[bos] > 0 and (af[bos] / as_[bos]) > VOL_MAX_RATIO:
                break

            start  = next((ph for ph in reversed(ph_idx_list) if ph < pl), max(0, pl - 30))
            segment = range(start, bos)

            ob_idx = max(segment, key=lambda i: highs[i], default=None)
            if ob_idx is None:
                break

            ob_h = highs[ob_idx]
            ob_l = lows[ob_idx]

            if any(closes[i] > ob_h for i in range(bos, n)):
                break

            obs.append(OrderBlock(
                direction  = "bear",
                ob_high    = ob_h,
                ob_low     = ob_l,
                ob_mid     = ob_h - (ob_h - ob_l) * OB_ENTRY_PCT,
                atr_at_bos = atrs[bos],
            ))
            break

    return obs


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 6 · MOTOR DE RIESGO
# ══════════════════════════════════════════════════════════════════════

def build_trade(
    ob: OrderBlock,
    equity: float,
    min_qty: float,
    qty_step: float,
    tick_size: float,
) -> Optional[Trade]:
    """
    Calcula todos los parámetros de la operación.
    Retorna None si no cumple con el R:R mínimo o la cantidad mínima.
    """
    entry = ob.ob_mid
    atr   = ob.atr_at_bos

    # Stop Loss: debajo del mínimo del OB + colchón ATR
    sl = (ob.ob_low - SL_ATR_PAD * atr) if ob.direction == "bull" else (ob.ob_high + SL_ATR_PAD * atr)

    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return None

    # Take Profit a 2.5R
    tp = (entry + sl_dist * MIN_RR) if ob.direction == "bull" else (entry - sl_dist * MIN_RR)

    # Tamaño de posición: riesgo fijo en dólares / distancia SL por unidad
    # XAUUSDT: 1 contrato = 1 oz de oro → distancia en USD/oz
    risk_usd = equity * RISK_PCT
    raw_qty  = risk_usd / sl_dist

    # Redondear al step permitido por el exchange
    qty = max(round(round(raw_qty / qty_step) * qty_step, 8), min_qty)

    def rp(p: float) -> float:
        """Redondear precio al tick_size del contrato."""
        return round(round(p / tick_size) * tick_size, 6)

    return Trade(
        direction   = ob.direction,
        entry_price = rp(entry),
        sl_price    = rp(sl),
        tp_price    = rp(tp),
        qty         = qty,
        risk_usd    = risk_usd,
    )


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 7 · FILTROS DE PROTECCIÓN
# ══════════════════════════════════════════════════════════════════════

# Lista de eventos de alto impacto (se pueden agregar en tiempo real
# o cargar desde un calendario externo)
NEWS_EVENTS: List[datetime] = []


def session_active() -> bool:
    """Solo opera durante el solapamiento Londres–Nueva York (13–17 GMT)."""
    h = datetime.now(timezone.utc).hour
    return SESSION_START <= h < SESSION_END


def market_is_trending(df: pd.DataFrame) -> bool:
    """ADX > umbral mínimo Y rango de últimas N velas > 2×ATR."""
    last = df.iloc[-1]
    adx_ok    = last["adx"] >= ADX_MIN
    rng       = df["high"].tail(ADX_LATERAL_N).max() - df["low"].tail(ADX_LATERAL_N).min()
    range_ok  = rng >= 2 * last["atr"]
    return adx_ok and range_ok


def volatility_normal(df: pd.DataFrame) -> bool:
    """ATR rápido no supera 2.5× el ATR lento."""
    last = df.iloc[-1]
    if last["atr_slow"] == 0:
        return True
    return (last["atr_fast"] / last["atr_slow"]) <= VOL_MAX_RATIO


def in_news_window() -> bool:
    """Pausa NEWS_PAUSE_MIN minutos alrededor de eventos macro."""
    now = datetime.now(timezone.utc)
    margin = timedelta(minutes=NEWS_PAUSE_MIN)
    return any(abs(now - e) <= margin for e in NEWS_EVENTS)


def spread_acceptable(ob: dict, history: List[float]) -> bool:
    """
    Spread actual <= percentil 95 del historial.
    Protege contra manipulación y baja liquidez.
    """
    try:
        bid = ob["bids"][0][0]
        ask = ob["asks"][0][0]
        sp  = ask - bid
        history.append(sp)
        if len(history) > SPREAD_HIST_N:
            history.pop(0)
        if len(history) < 10:
            return True
        return sp <= float(np.percentile(history, 95))
    except Exception:
        return True  # Si no hay datos, no bloquear


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 8 · RASTREADOR DE RIESGO DIARIO / SEMANAL
# ══════════════════════════════════════════════════════════════════════

def refresh_risk_tracker(rt: RiskTracker, equity: float):
    """Resetea los contadores en nuevos días y semanas."""
    now   = datetime.now(timezone.utc)
    day   = now.strftime("%Y-%m-%d")
    week  = now.strftime("%Y-W%W")

    if rt.reset_day != day:
        rt.equity_day = equity
        rt.pnl_day    = 0.0
        rt.reset_day  = day
        log.info(f"Reset diario | Equity: ${equity:.2f}")

    if rt.reset_week != week:
        rt.equity_week = equity
        rt.pnl_week    = 0.0
        rt.reset_week  = week
        log.info(f"Reset semanal | Equity: ${equity:.2f}")


def risk_limit_hit(rt: RiskTracker) -> Tuple[bool, str]:
    """Retorna (violado, motivo) para los límites de pérdida."""
    if rt.equity_day > 0:
        daily = rt.pnl_day / rt.equity_day
        if daily <= -DAILY_LOSS_LIM:
            return True, f"Límite diario: {daily*100:.1f}% (máx −{DAILY_LOSS_LIM*100:.0f}%)"

    if rt.equity_week > 0:
        weekly = rt.pnl_week / rt.equity_week
        if weekly <= -WEEKLY_LOSS_LIM:
            return True, f"Límite semanal: {weekly*100:.1f}% (máx −{WEEKLY_LOSS_LIM*100:.0f}%)"

    return False, ""


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 9 · CLASE PRINCIPAL DEL BOT
# ══════════════════════════════════════════════════════════════════════

class OBBot:
    """
    Máquina de estados asíncrona que implementa la estrategia de
    Order Blocks sobre XAUUSDT Perpetual Futures en Binance.

    Estados:
        IDLE    → busca setups en cada vela cerrada
        PENDING → orden límite enviada, vigila fill e invalidaciones
        ACTIVE  → posición abierta, gestiona SL/TP y break-even
        STOPPED → límite de pérdida alcanzado, hiberna hasta reset
    """

    def __init__(self):
        self.exchange = ccxt.binance({
            "apiKey":    API_KEY,
            "secret":    API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType":              "future",
                "adjustForTimeDifference":  True,
            },
        })

        self.state          : BotState          = BotState.IDLE
        self.trade          : Optional[Trade]   = None
        self.pending_ob     : Optional[OrderBlock] = None
        self.risk           : RiskTracker       = RiskTracker()
        self.spread_hist    : List[float]       = []

        # Metadatos del contrato (cargados al inicio)
        self.min_qty   : float = 0.01
        self.qty_step  : float = 0.01
        self.tick_size : float = 0.01

        self._last_candle_ts = None   # Para detectar nuevas velas cerradas

    # ── Capa de red con exponential backoff ───────────────────────────

    async def _call(self, coro, retries: int = MAX_RETRIES):
        for attempt in range(retries):
            try:
                return await coro
            except ccxt.RateLimitExceeded:
                wait = RETRY_BASE * (2 ** attempt)
                log.warning(f"Rate limit — esperando {wait:.0f}s")
                await asyncio.sleep(wait)
            except ccxt.NetworkError as e:
                wait = RETRY_BASE * (2 ** attempt)
                log.warning(f"Red: {e} — reintento {attempt+1}/{retries} en {wait:.0f}s")
                await asyncio.sleep(wait)
            except ccxt.ExchangeError as e:
                log.error(f"Exchange: {e}")
                raise
        raise RuntimeError("Máximo de reintentos alcanzado.")

    # ── Utilidades del exchange ───────────────────────────────────────

    async def load_contract(self):
        """Carga metadatos del contrato una sola vez al inicio."""
        markets = await self._call(self.exchange.load_markets())
        if CCXT_SYMBOL not in markets:
            raise ValueError(f"Símbolo {CCXT_SYMBOL} no disponible.")
        m = markets[CCXT_SYMBOL]

        def to_float(v):
            return v if isinstance(v, float) else (10 ** -v if isinstance(v, int) else 0.01)

        self.min_qty   = float(m["limits"]["amount"]["min"] or 0.01)
        self.tick_size = to_float(m["precision"]["price"])
        self.qty_step  = to_float(m["precision"]["amount"])
        log.info(f"Contrato: min_qty={self.min_qty} | tick={self.tick_size} | step={self.qty_step}")

    async def equity(self) -> float:
        """Equity total incluyendo PnL no realizado."""
        bal = await self._call(self.exchange.fetch_balance())
        return float(bal.get("total", {}).get("USDT", 0))

    async def get_position(self) -> Optional[dict]:
        """Posición abierta en XAUUSDT, o None."""
        positions = await self._call(self.exchange.fetch_positions([CCXT_SYMBOL]))
        for p in positions:
            if abs(float(p.get("contracts", 0))) > 0:
                return p
        return None

    async def cancel_all(self):
        try:
            await self._call(self.exchange.cancel_all_orders(CCXT_SYMBOL))
            log.info("Órdenes canceladas.")
        except Exception as e:
            log.error(f"cancel_all: {e}")

    async def close_position(self):
        """Cierre de emergencia con orden de mercado."""
        pos = await self.get_position()
        if not pos:
            return
        contracts = float(pos["contracts"])
        side = "sell" if contracts > 0 else "buy"
        await self._call(
            self.exchange.create_order(
                CCXT_SYMBOL, "market", side, abs(contracts),
                params={"reduceOnly": True}
            )
        )
        log.info(f"Posición cerrada @ mercado: {abs(contracts)} oz")

    async def candles(self) -> pd.DataFrame:
        """Descarga velas, excluye la activa y enriquece con indicadores."""
        raw = await self._call(
            self.exchange.fetch_ohlcv(CCXT_SYMBOL, TIMEFRAME, limit=CANDLE_LIMIT)
        )
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df = df.astype(float).iloc[:-1]   # ← excluir vela incompleta
        return enrich_dataframe(df)

    # ── Colocación de órdenes ─────────────────────────────────────────

    async def send_entry_order(self, t: Trade) -> Optional[str]:
        side = "buy" if t.direction == "bull" else "sell"
        order = await self._call(
            self.exchange.create_limit_order(CCXT_SYMBOL, side, t.qty, t.entry_price)
        )
        log.info(f"Entrada enviada | {side.upper()} {t.qty} oz @ {t.entry_price}")
        return order["id"]

    async def send_sl_tp(self, t: Trade):
        """Coloca SL (stop-market) y TP (take-profit-market) tras el fill."""
        side_close = "sell" if t.direction == "bull" else "buy"

        sl_order = await self._call(
            self.exchange.create_order(
                CCXT_SYMBOL, "stop_market", side_close, t.qty,
                params={"stopPrice": t.sl_price, "reduceOnly": True},
            )
        )
        t.sl_order_id = sl_order["id"]

        tp_order = await self._call(
            self.exchange.create_order(
                CCXT_SYMBOL, "take_profit_market", side_close, t.qty,
                params={"stopPrice": t.tp_price, "reduceOnly": True},
            )
        )
        t.tp_order_id = tp_order["id"]
        log.info(f"SL: {t.sl_price} | TP: {t.tp_price}")

    async def move_sl_to_be(self, t: Trade):
        """Break-even: reemplaza el SL por uno en el precio de entrada."""
        if t.be_triggered or not t.sl_order_id:
            return
        try:
            await self._call(self.exchange.cancel_order(t.sl_order_id, CCXT_SYMBOL))
            side_close = "sell" if t.direction == "bull" else "buy"
            new_sl = await self._call(
                self.exchange.create_order(
                    CCXT_SYMBOL, "stop_market", side_close, t.qty,
                    params={"stopPrice": t.entry_price, "reduceOnly": True},
                )
            )
            t.sl_order_id  = new_sl["id"]
            t.sl_price     = t.entry_price
            t.be_triggered = True
            log.info(f"Break-even en {t.entry_price}")
            tg(f"🔒 <b>BREAK-EVEN</b> activado en {t.entry_price}")
        except Exception as e:
            log.error(f"BE error: {e}")

    # ── Máquina de estados ────────────────────────────────────────────

    async def on_idle(self, df: pd.DataFrame):
        """Busca setups cuando no hay posición activa."""
        # ── Filtros globales ──────────────────────────────────────────
        if not session_active():
            return
        if not market_is_trending(df):
            log.debug("Mercado lateral → sin operaciones.")
            return
        if not volatility_normal(df):
            log.warning("Volatilidad extrema → bloqueado.")
            return
        if in_news_window():
            log.warning("Ventana de noticias → bloqueado.")
            return

        # ── Detectar Order Blocks ─────────────────────────────────────
        obs = [ob for ob in find_order_blocks(df) if ob.valid and ob.retests <= MAX_OB_RETESTS]
        if not obs:
            return

        ob = obs[-1]   # El OB más reciente es el de mayor probabilidad
        eq = await self.equity()

        trade = build_trade(ob, eq, self.min_qty, self.qty_step, self.tick_size)
        if trade is None:
            log.warning("Trade descartado: parámetros inválidos (R:R insuficiente o qty < mínimo).")
            return

        oid = await self.send_entry_order(trade)
        if oid:
            trade.entry_order_id = oid
            self.trade      = trade
            self.pending_ob = ob
            self.state      = BotState.PENDING

            direction_label = "LONG 🟢" if trade.direction == "bull" else "SHORT 🔴"
            tg(
                f"⏳ <b>ORDEN COLOCADA</b>\n"
                f"📍 {direction_label} | {trade.qty} oz @ <b>{trade.entry_price}</b>\n"
                f"🛑 SL: {trade.sl_price}  🎯 TP: {trade.tp_price}\n"
                f"⚖️ R:R = 1:{MIN_RR}  |  Riesgo: <b>${trade.risk_usd:.2f}</b>"
            )

    async def on_pending(self, df: pd.DataFrame):
        """Vigila si la orden se ejecutó o si el OB se invalidó."""
        if not self.trade or not self.trade.entry_order_id:
            self.state = BotState.IDLE
            return

        # ── Verificar invalidación del OB ─────────────────────────────
        ob   = self.pending_ob
        last = df.iloc[-1]["close"]

        if ob:
            invalid = (ob.direction == "bull" and last < ob.ob_low) or \
                      (ob.direction == "bear" and last > ob.ob_high)
            if invalid:
                await self.cancel_all()
                self.trade, self.pending_ob = None, None
                self.state = BotState.IDLE
                tg("❌ <b>ORDEN CANCELADA</b> — OB invalidado por precio.")
                return

        # ── Verificar si la orden fue ejecutada ───────────────────────
        try:
            order = await self._call(
                self.exchange.fetch_order(self.trade.entry_order_id, CCXT_SYMBOL)
            )
            if order["status"] == "closed":
                self.trade.entry_price = float(order.get("average") or order["price"])
                await self.send_sl_tp(self.trade)
                self.state = BotState.ACTIVE

                direction_label = "LONG 🟢" if self.trade.direction == "bull" else "SHORT 🔴"
                tg(
                    f"✅ <b>TRADE ACTIVO</b>\n"
                    f"📍 {direction_label} {self.trade.qty} oz @ <b>{self.trade.entry_price}</b>\n"
                    f"🛑 SL: {self.trade.sl_price}  🎯 TP: {self.trade.tp_price}"
                )
        except Exception as e:
            log.error(f"on_pending fetch_order: {e}")

    async def on_active(self, df: pd.DataFrame):
        """Gestiona la posición: break-even y detección de cierre."""
        if not self.trade:
            self.state = BotState.IDLE
            return

        # ── Verificar si la posición sigue abierta ────────────────────
        pos = await self.get_position()
        if not pos:
            await self._on_close()
            return

        t     = self.trade
        price = df.iloc[-1]["close"]
        risk  = abs(t.entry_price - t.sl_price)

        # ── Break-even al alcanzar 1R de ganancia flotante ────────────
        if not t.be_triggered:
            be_trigger = (
                (t.direction == "bull" and price >= t.entry_price + risk) or
                (t.direction == "bear" and price <= t.entry_price - risk)
            )
            if be_trigger:
                await self.move_sl_to_be(t)

    async def _on_close(self):
        """Limpia estado y registra PnL cuando se cierra una posición."""
        pnl = 0.0
        try:
            recent_trades = await self._call(
                self.exchange.fetch_my_trades(CCXT_SYMBOL, limit=10)
            )
            pnl = sum(float(t.get("realizedPnl", 0)) for t in recent_trades[-4:])
        except Exception as e:
            log.error(f"_on_close: {e}")

        self.risk.pnl_day  += pnl
        self.risk.pnl_week += pnl

        icon = "✅" if pnl >= 0 else "❌"
        tg(f"{icon} <b>TRADE CERRADO</b>\n💵 PnL realizado: <b>${pnl:.2f}</b>")
        log.info(f"Trade cerrado. PnL: {pnl:.2f}")

        await self.cancel_all()
        self.trade, self.pending_ob = None, None
        self.state = BotState.IDLE

    # ── Loop principal ─────────────────────────────────────────────────

    async def run(self):
        log.info("═" * 65)
        log.info("  XAUUSDT ORDER BLOCK BOT  ·  INICIANDO")
        log.info("═" * 65)

        await self.load_contract()

        # Configurar One-Way Mode y apalancamiento inicial
        try:
            await self._call(self.exchange.set_position_mode(False))
            await self._call(self.exchange.set_leverage(LEVERAGE, CCXT_SYMBOL))
        except Exception as e:
            log.warning(f"Configuración inicial: {e}")

        # Cerrar posiciones residuales de sesiones anteriores
        if await self.get_position():
            log.warning("Posición residual encontrada — cerrando...")
            await self.close_position()
        await self.cancel_all()

        eq = await self.equity()
        tg(
            f"🟢 <b>BOT INICIADO</b>\n"
            f"💰 Capital: <b>${eq:.2f}</b>\n"
            f"📊 {CCXT_SYMBOL} | {TIMEFRAME}\n"
            f"⚠️ Riesgo/trade: {RISK_PCT*100:.0f}%  |  R:R: 1:{MIN_RR}\n"
            f"🕐 Sesión: {SESSION_START}:00–{SESSION_END}:00 GMT"
        )

        while True:
            try:
                # ── Actualizar equity y límites de pérdida ─────────────
                eq = await self.equity()
                refresh_risk_tracker(self.risk, eq)

                breached, reason = risk_limit_hit(self.risk)
                if breached and self.state != BotState.STOPPED:
                    log.warning(f"LÍMITE: {reason}")
                    tg(f"🛑 <b>LÍMITE DE PÉRDIDA</b>\n{reason}\nBot en pausa hasta reset.")
                    await self.cancel_all()
                    await self.close_position()
                    self.trade, self.pending_ob = None, None
                    self.state = BotState.STOPPED

                if self.state == BotState.STOPPED:
                    # Verificar si es un nuevo día/semana para reactivarse
                    _, reason2 = risk_limit_hit(self.risk)
                    if not reason2:
                        self.state = BotState.IDLE
                        tg("🟡 <b>REACTIVADO</b> — Nuevo período de riesgo.")
                    await asyncio.sleep(60)
                    continue

                # ── Descargar datos ────────────────────────────────────
                df = await self.candles()

                # ── Filtro de spread ───────────────────────────────────
                spread_ok = True
                try:
                    ob_data   = await self._call(self.exchange.fetch_order_book(CCXT_SYMBOL, limit=5))
                    spread_ok = spread_acceptable(ob_data, self.spread_hist)
                    if not spread_ok:
                        log.warning("Spread anómalo — ciclo saltado.")
                except Exception:
                    pass

                # ── Máquina de estados ─────────────────────────────────
                last_ts = df.index[-1]

                if self.state == BotState.IDLE and spread_ok:
                    if last_ts != self._last_candle_ts:
                        self._last_candle_ts = last_ts
                        await self.on_idle(df)

                elif self.state == BotState.PENDING:
                    await self.on_pending(df)

                elif self.state == BotState.ACTIVE:
                    await self.on_active(df)

                await asyncio.sleep(LOOP_SEC)

            except ccxt.NetworkError as e:
                log.error(f"Red: {e} — reconectando en 30s")
                await asyncio.sleep(30)
            except ccxt.ExchangeError as e:
                log.error(f"Exchange: {e}")
                await asyncio.sleep(15)
            except Exception as e:
                log.error(f"Error inesperado: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def shutdown(self):
        await self.exchange.close()


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 10 · ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

async def main():
    bot = OBBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        log.info("Apagado manual.")
        tg("🔴 <b>Bot detenido manualmente.</b>")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
