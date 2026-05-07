"""
╔══════════════════════════════════════════════════════════════════════╗
║   XAUUSDT EMA CROSSOVER BOT  ·  v3.1                                ║
║   Estrategia : EMA 7 / EMA 14 Crossover  ·  Marco: M15             ║
║   Exchange   : Binance Futures USDT-M  ·  One-Way Mode              ║
║   Lógica     : Siempre en mercado — Long o Short según cruce        ║
╚══════════════════════════════════════════════════════════════════════╝

  GATILLOS:
    EMA7 cruza SOBRE EMA14  → cerrar SHORT (si existe) + abrir LONG
    EMA7 cruza BAJO  EMA14  → cerrar LONG  (si existe) + abrir SHORT
  El ciclo se repite indefinidamente sin filtros de tendencia macro.

  BAN DE IP:
    Si PROXY_LIST está configurado, el bot rota de proxy automáticamente.
    Sin proxies, espera el tiempo del ban y continúa solo.
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
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 · CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════
load_dotenv()

# ── Credenciales (cargadas desde .env) ────────────────────────────────
API_KEY    = os.getenv("API_KEY",    "")
API_SECRET = os.getenv("API_SECRET", "")
TG_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TG_CHAT    = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Instrumento ───────────────────────────────────────────────────────
CCXT_SYMBOL = "XAU/USDT:USDT"   # Binance Futures USDT-M
TIMEFRAME   = "15m"
LEVERAGE    = 1

# ── Estrategia EMA ────────────────────────────────────────────────────
EMA_FAST = 7    # EMA rápida
EMA_SLOW = 14   # EMA lenta

# ── Proxies (opcional — rotación automática si Binance banea la IP) ───
# En .env:  PROXY_LIST=http://user:pass@host1:port,http://user:pass@host2:port
PROXY_LIST_RAW = os.getenv("PROXY_LIST", "")

# ── Tamaño de posición ────────────────────────────────────────────────
# Porcentaje del equity que se usa como margen por operación.
# Con LEVERAGE=1 y MARGIN_PCT=0.95 → 95% del equity como nocional (sin apalancamiento).
# Se deja 5% de buffer para fees y variaciones de precio.
MARGIN_PCT = 0.95   # 95% del equity — sin apalancamiento

# ── Infraestructura ───────────────────────────────────────────────────
CANDLE_LIMIT = 60       # Velas a descargar (más que suficiente para EMA7/14)
LOOP_SEC     = 10       # Segundos entre ciclos del loop principal
MAX_RETRIES  = 5        # Reintentos para llamadas API
RETRY_BASE   = 2.0      # Base del exponential backoff (segundos)
CLOSE_VERIFY_RETRIES = 4  # Intentos para confirmar cierre de posición

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("EMABot")


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 · TELEGRAM
# ══════════════════════════════════════════════════════════════════════

def tg(msg: str):
    """Envía notificación a Telegram."""
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram no configurado: TG_TOKEN o TG_CHAT vacíos.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.ok:
            log.warning(f"Telegram error {r.status_code}: {r.text}")
    except Exception as e:
        log.warning(f"Telegram excepción: {e}")


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 · INDICADORES
# ══════════════════════════════════════════════════════════════════════

def calc_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


# ══════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 · BOT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class EMABot:
    """
    Bot de cruce de EMAs para XAUUSDT Perpetual Futures en Binance.

    Siempre mantiene una posición activa (Long o Short) según el cruce
    de EMA7 sobre EMA14. Cuando ocurre un cruce:
      1. Cancela todas las órdenes pendientes.
      2. Cierra la posición actual con orden de mercado (verificado).
      3. Abre inmediatamente la posición contraria con orden de mercado.
    """

    def __init__(self):
        # Lista de proxies para rotación automática ante bans de IP
        self._proxy_list: list[str] = [
            p.strip() for p in PROXY_LIST_RAW.split(",") if p.strip()
        ]
        self._proxy_idx: int = 0

        self._init_exchange(self._proxy_list[0] if self._proxy_list else None)

        # Estado interno — siempre sincronizado con el exchange antes de operar
        self.current_side: Optional[str] = None   # "long", "short" o None
        self.current_qty:  float = 0.0

        # Metadatos del contrato
        self.min_qty:   float = 0.01
        self.qty_step:  float = 0.01
        self.tick_size: float = 0.01

        # Timestamp de la última vela procesada (evita doble señal)
        self._last_candle_ts = None

        # Equity al abrir la posición actual — usado para calcular PnL real
        self._entry_equity: float = 0.0

    def _init_exchange(self, proxy: Optional[str] = None):
        """Crea (o recrea) la instancia del exchange, opcionalmente con proxy."""
        config: dict = {
            "apiKey":          API_KEY,
            "secret":          API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType":             "future",
                "adjustForTimeDifference": True,
            },
        }
        if proxy:
            config["aiohttp_proxy"] = proxy
        self.exchange = ccxt.binance(config)

    # ── Capa de red con exponential backoff y rotación de proxy ──────

    def _is_ban_error(self, error_msg: str) -> bool:
        return "banned" in error_msg.lower() or "418" in error_msg

    def _ban_remaining(self, error_msg: str) -> float:
        """Segundos restantes del ban según el timestamp de Binance, o 60 si no hay."""
        import re, time
        match = re.search(r'"msg".*?until\s+(\d+)', error_msg)
        if not match:
            return 60.0
        remaining = (int(match.group(1)) / 1000) - time.time()
        return max(remaining, 1.0)

    async def _rotate_proxy(self) -> bool:
        """
        Rota al siguiente proxy disponible y recrea el exchange.
        Retorna True si la rotación fue exitosa, False si no hay proxies.
        """
        if not self._proxy_list:
            return False

        old_exchange = self.exchange
        self._proxy_idx = (self._proxy_idx + 1) % len(self._proxy_list)
        proxy = self._proxy_list[self._proxy_idx]

        log.warning(f"Rotando proxy → {proxy} ({self._proxy_idx + 1}/{len(self._proxy_list)})")
        self._init_exchange(proxy)

        try:
            await old_exchange.close()
        except Exception:
            pass

        tg(
            f"🔄 <b>Proxy rotado automáticamente</b>\n"
            f"🌐 Proxy {self._proxy_idx + 1}/{len(self._proxy_list)} activo\n"
            f"🤖 Continuando sin intervención manual..."
        )
        return True

    async def _handle_ban(self, error_msg: str) -> float:
        """
        Maneja un ban de IP de Binance.
        - Si hay proxies: rota y retorna 5s para reintentar rápido.
        - Si no hay proxies: espera el tiempo del ban automáticamente.
        Retorna 0 si el error no es un ban.
        """
        if not self._is_ban_error(error_msg):
            return 0.0

        remaining = self._ban_remaining(error_msg)
        mins = remaining / 60
        log.warning(f"IP baneada por Binance — ban expira en {mins:.1f} minutos.")

        rotated = await self._rotate_proxy()
        if rotated:
            return 5.0  # Reintentar rápido con el proxy nuevo

        # Sin proxies: esperar automáticamente en bloques de 5 min
        tg(
            f"🚫 <b>IP baneada por Binance</b>\n"
            f"⏳ Ban expira en: <b>{mins:.0f} minutos</b>\n"
            f"⏸️ Esperando automáticamente — no se requiere intervención."
        )
        return min(remaining, 300.0)

    async def _call(self, fn, retries: int = MAX_RETRIES):
        """
        Llama a fn() con reintentos y backoff exponencial.
        fn debe ser un callable (lambda) que retorne una coroutine,
        así se crea una coroutine fresca en cada reintento.
        """
        for attempt in range(retries):
            try:
                return await fn()
            except ccxt.RateLimitExceeded as e:
                wait = await self._handle_ban(str(e)) or RETRY_BASE * (2 ** attempt)
                log.warning(f"Rate limit — esperando {wait:.0f}s")
                await asyncio.sleep(wait)
            except ccxt.NetworkError as e:
                wait = await self._handle_ban(str(e))
                if wait:
                    await asyncio.sleep(wait)
                else:
                    wait = RETRY_BASE * (2 ** attempt)
                    log.warning(f"Red: {e} — reintento {attempt+1}/{retries} en {wait:.0f}s")
                    await asyncio.sleep(wait)
            except ccxt.ExchangeError as e:
                log.error(f"Exchange: {e}")
                raise
        raise RuntimeError("Máximo de reintentos alcanzado.")

    # ── Utilidades del contrato ───────────────────────────────────────

    async def load_contract(self):
        markets = await self._call(lambda: self.exchange.load_markets())
        if CCXT_SYMBOL not in markets:
            raise ValueError(f"Símbolo {CCXT_SYMBOL} no disponible.")
        m = markets[CCXT_SYMBOL]

        def to_float(v):
            return v if isinstance(v, float) else (10 ** -v if isinstance(v, int) else 0.01)

        self.min_qty   = float(m["limits"]["amount"]["min"] or 0.01)
        self.tick_size = to_float(m["precision"]["price"])
        self.qty_step  = to_float(m["precision"]["amount"])
        log.info(f"Contrato: min_qty={self.min_qty} | tick={self.tick_size} | step={self.qty_step}")

    def _round_qty(self, qty: float) -> float:
        return max(round(round(qty / self.qty_step) * self.qty_step, 8), self.min_qty)

    def _round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 6)

    async def _equity(self) -> float:
        bal = await self._call(lambda: self.exchange.fetch_balance())
        return float(bal.get("total", {}).get("USDT", 0))

    # ── Posición real en el exchange ──────────────────────────────────

    async def _get_position(self) -> Optional[dict]:
        """Retorna la posición abierta en el exchange o None."""
        positions = await self._call(lambda: self.exchange.fetch_positions([CCXT_SYMBOL]))
        for p in positions:
            if abs(float(p.get("contracts", 0))) > 0:
                return p
        return None

    async def sync_state(self):
        """
        Reconcilia el estado interno con la posición real en el exchange.
        Llamar siempre antes de tomar decisiones de trading.
        """
        pos = await self._get_position()
        if not pos:
            if self.current_side is not None:
                log.warning("Posición desincronizada — ajustando estado interno a: ninguna")
            self.current_side = None
            self.current_qty  = 0.0
        else:
            contracts = float(pos.get("contracts", 0))
            # En Binance one-way mode, contracts es SIEMPRE positivo.
            # Usar el campo "side" para saber la dirección real.
            pos_side = pos.get("side", "")   # "long" o "short" según ccxt

            if abs(contracts) < self.min_qty:
                new_side = None
                new_qty  = 0.0
            elif pos_side == "short":
                new_side = "short"
                new_qty  = abs(contracts)
            else:
                # "long" o cualquier valor desconocido → tratar como long
                new_side = "long"
                new_qty  = abs(contracts)

            if new_side != self.current_side:
                log.info(f"Sincronización: estado actualizado a {new_side} ({new_qty} oz) [side={pos_side}]")
            self.current_side = new_side
            self.current_qty  = new_qty

    # ── Notificación de cierre ────────────────────────────────────────

    async def _notify_close(self, closed_side: str, qty: float):
        """
        Calcula el PnL real como diferencia de equity (entrada vs. cierre)
        y notifica por Telegram. Este método es preciso porque no depende
        de sumar fills individuales — usa directamente el equity del exchange.
        """
        equity = 0.0
        try:
            equity = await self._equity()
        except Exception as e:
            log.warning(f"No se pudo obtener equity: {e}")

        # PnL = equity al cerrar − equity al abrir (incluye fees y funding)
        pnl = (equity - self._entry_equity) if self._entry_equity > 0 else 0.0

        icon_side  = "🟢" if closed_side == "long" else "🔴"
        icon_pnl   = "✅" if pnl >= 0 else "❌"
        label_side = "LONG" if closed_side == "long" else "SHORT"
        pnl_str    = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        log.info(f"Cierre {label_side} | PnL: {pnl_str} | Equity: ${equity:.2f}")
        tg(
            f"{icon_side} <b>{label_side} CERRADO</b>\n"
            f"{icon_pnl} Resultado: <b>{pnl_str}</b>\n"
            f"💰 Capital actual: <b>${equity:.2f}</b>"
        )

    # ── Cierre de posición (crítico — verificado) ─────────────────────

    async def close_position_market(self) -> bool:
        """
        Cierra la posición actual con orden de mercado.

        Proceso:
          1. Cancela todas las órdenes pendientes.
          2. Obtiene la cantidad exacta de la posición desde el exchange.
          3. Envía orden de mercado con reduceOnly=True.
          4. Verifica que la posición se cerró realmente (hasta N reintentos).

        Retorna True si el cierre fue exitoso, False si falló tras reintentos.
        """
        # 1. Cancelar órdenes pendientes primero para evitar conflictos
        try:
            await self._call(lambda: self.exchange.cancel_all_orders(CCXT_SYMBOL))
            log.info("Órdenes pendientes canceladas.")
        except Exception as e:
            log.warning(f"cancel_all: {e}")

        # 2. Verificar posición real en el exchange
        pos = await self._get_position()
        if not pos:
            log.info("No hay posición abierta que cerrar.")
            self.current_side = None
            self.current_qty  = 0.0
            return True

        contracts = float(pos.get("contracts", 0))
        pos_side  = pos.get("side", "")   # "long" o "short" en ccxt / Binance

        if abs(contracts) < self.min_qty:
            log.info("Posición despreciable, considerada cerrada.")
            self.current_side = None
            self.current_qty  = 0.0
            return True

        # En Binance one-way mode contracts es siempre positivo.
        # Usar "side" para determinar la dirección correcta del cierre.
        if pos_side == "short":
            close_side = "buy"    # Cerrar short → comprar
        else:
            close_side = "sell"   # Cerrar long  → vender

        qty_to_close = abs(contracts)
        log.info(f"Cerrando posición: {pos_side.upper()} {qty_to_close} oz → {close_side.upper()} @ mercado (reduceOnly)")

        # 3 + 4. Enviar y verificar hasta CLOSE_VERIFY_RETRIES intentos
        for attempt in range(1, CLOSE_VERIFY_RETRIES + 1):
            try:
                await self._call(
                    lambda: self.exchange.create_order(
                        CCXT_SYMBOL,
                        "market",
                        close_side,
                        qty_to_close,
                        params={"reduceOnly": True},
                    )
                )
                log.info(f"Orden de cierre enviada (intento {attempt}).")
            except Exception as e:
                log.error(f"Error enviando orden de cierre (intento {attempt}): {e}")
                await asyncio.sleep(2)
                continue

            # Esperar a que el exchange procese la orden
            await asyncio.sleep(1.5)

            # Verificar que la posición se cerró
            pos_after = await self._get_position()
            if not pos_after or abs(float(pos_after.get("contracts", 0))) < self.min_qty:
                log.info("Posición cerrada y verificada correctamente.")
                self.current_side = None
                self.current_qty  = 0.0
                await self._notify_close(pos_side, qty_to_close)
                return True

            remaining = abs(float(pos_after.get("contracts", 0)))
            log.warning(
                f"Posición aún abierta tras intento {attempt}: {remaining} oz. "
                f"Reintentando..."
            )
            qty_to_close = remaining  # Actualizar por si fue cierre parcial
            await asyncio.sleep(2)

        log.error(f"No se pudo cerrar la posición tras {CLOSE_VERIFY_RETRIES} intentos.")
        tg(f"⚠️ <b>ERROR CRÍTICO</b>: No se pudo cerrar la posición tras {CLOSE_VERIFY_RETRIES} intentos. Revisar manualmente.")
        return False

    # ── Apertura de posición ──────────────────────────────────────────

    async def open_position_market(self, side: str, ref_price: float) -> bool:
        """
        Abre una posición de mercado.
        side: 'long' o 'short'
        ref_price: precio de referencia para calcular la cantidad (close de la última vela)
        """
        eq = await self._equity()
        self._entry_equity = eq   # guardar equity de entrada para PnL correcto
        notional = eq * MARGIN_PCT * LEVERAGE
        qty = self._round_qty(notional / ref_price)

        exchange_side = "buy" if side == "long" else "sell"
        icon = "🟢" if side == "long" else "🔴"
        label = "LONG" if side == "long" else "SHORT"

        log.info(f"Abriendo {label}: {exchange_side.upper()} {qty} oz @ mercado")

        try:
            await self._call(
                lambda: self.exchange.create_order(
                    CCXT_SYMBOL,
                    "market",
                    exchange_side,
                    qty,
                )
            )
            self.current_side = side
            self.current_qty  = qty

            tg(
                f"{icon} <b>{label} ABIERTO</b>\n"
                f"📊 {qty} oz  |  Ref. precio: ~{self._round_price(ref_price)}\n"
                f"⚡ EMA{EMA_FAST} cruzó {'↑ sobre' if side == 'long' else '↓ bajo'} EMA{EMA_SLOW}"
            )
            log.info(f"{label} abierto: {qty} oz")
            return True
        except Exception as e:
            log.error(f"Error abriendo {label}: {e}")
            tg(f"⚠️ Error abriendo {label}: {e}")
            return False

    # ── Datos de mercado e indicadores ────────────────────────────────

    async def candles(self) -> pd.DataFrame:
        """Descarga velas, excluye la activa y calcula EMAs."""
        raw = await self._call(
            lambda: self.exchange.fetch_ohlcv(CCXT_SYMBOL, TIMEFRAME, limit=CANDLE_LIMIT)
        )
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df = df.astype(float)              # incluir vela activa para detección en tiempo real
        df["ema_fast"] = calc_ema(df["close"], EMA_FAST)
        df["ema_slow"] = calc_ema(df["close"], EMA_SLOW)
        return df

    # ── Detección de cruce ────────────────────────────────────────────

    def detect_crossover(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detecta cruce de EMA rápida (7) sobre EMA lenta (14).

        Usa las dos últimas velas CERRADAS para confirmar el cruce:
          - Vela anterior: EMA7 estaba por debajo de EMA14
          - Vela actual:   EMA7 está por encima de EMA14  → señal LONG

          - Vela anterior: EMA7 estaba por encima de EMA14
          - Vela actual:   EMA7 está por debajo de EMA14  → señal SHORT

        Retorna 'long', 'short' o None (sin cruce).
        """
        if len(df) < 3:
            return None

        prev_fast = df["ema_fast"].iloc[-2]
        prev_slow = df["ema_slow"].iloc[-2]
        curr_fast = df["ema_fast"].iloc[-1]
        curr_slow = df["ema_slow"].iloc[-1]

        crossed_up   = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if crossed_up:
            return "long"
        if crossed_down:
            return "short"
        return None

    # ── Loop principal ────────────────────────────────────────────────

    async def run(self):
        log.info("═" * 65)
        log.info(f"  XAUUSDT EMA{EMA_FAST}/{EMA_SLOW} CROSSOVER BOT  ·  INICIANDO")
        log.info("═" * 65)
        log.info(f"Telegram TOKEN configurado: {'SI' if TG_TOKEN else 'NO'}")
        log.info(f"Telegram CHAT_ID configurado: {'SI' if TG_CHAT else 'NO'}")

        await self.load_contract()

        # Configurar modo one-way y apalancamiento
        try:
            await self._call(lambda: self.exchange.set_position_mode(False))
            log.info("One-Way Mode configurado.")
        except Exception as e:
            log.warning(f"set_position_mode: {e} (puede ya estar configurado)")

        try:
            await self._call(lambda: self.exchange.set_leverage(LEVERAGE, CCXT_SYMBOL))
            log.info(f"Apalancamiento x{LEVERAGE} configurado.")
        except Exception as e:
            log.warning(f"set_leverage: {e}")

        # Sincronizar con el estado real del exchange al arrancar
        await self.sync_state()

        eq = await self._equity()
        if self.current_side:
            self._entry_equity = eq   # referencia de entrada para posición heredada
            log.info(f"Posición residual encontrada: {self.current_side.upper()} {self.current_qty} oz — se mantendrá.")
        tg(
            f"🟢 <b>EMA CROSSOVER BOT INICIADO</b>\n"
            f"💰 Equity: <b>${eq:.2f}</b>\n"
            f"📊 {CCXT_SYMBOL} | {TIMEFRAME}\n"
            f"📈 EMA{EMA_FAST} / EMA{EMA_SLOW} Crossover — siempre en mercado\n"
            f"⚙️ Margen/op: {MARGIN_PCT*100:.0f}% | Apalancamiento: x{LEVERAGE}\n"
            f"📍 Posición actual: {self.current_side.upper() if self.current_side else 'ninguna'}"
        )

        while True:
            try:
                df = await self.candles()
                last_ts = df.index[-1]

                ema_fast_val = float(df["ema_fast"].iloc[-1])
                ema_slow_val = float(df["ema_slow"].iloc[-1])
                ref_price    = float(df["close"].iloc[-1])

                # Log informativo solo cuando abre una nueva vela (cada 15 min)
                if last_ts != self._last_candle_ts:
                    self._last_candle_ts = last_ts
                    log.info(
                        f"Nueva vela: {last_ts} | "
                        f"Precio={ref_price:.2f} | "
                        f"EMA{EMA_FAST}={ema_fast_val:.2f} | "
                        f"EMA{EMA_SLOW}={ema_slow_val:.2f} | "
                        f"Posición: {self.current_side or 'ninguna'}"
                    )

                signal = self.detect_crossover(df)

                if signal is None:
                    log.debug("Sin cruce en esta vela.")
                    await asyncio.sleep(LOOP_SEC)
                    continue

                log.info(f"CRUCE DETECTADO → señal: {signal.upper()}")

                # Sincronizar estado con el exchange antes de operar
                await self.sync_state()

                # Si ya estamos en la dirección de la señal, nada que hacer
                if self.current_side == signal:
                    log.info(f"Ya en posición {signal.upper()}, señal ignorada.")
                    await asyncio.sleep(LOOP_SEC)
                    continue

                # ── EJECUTAR GIRO ──────────────────────────────────────
                # 1. Cerrar posición actual (si existe)
                if self.current_side is not None:
                    old_label = self.current_side.upper()
                    new_label = signal.upper()
                    tg(
                        f"🔄 <b>GIRO: {old_label} → {new_label}</b>\n"
                        f"EMA{EMA_FAST}={ema_fast_val:.2f} | EMA{EMA_SLOW}={ema_slow_val:.2f}"
                    )
                    closed = await self.close_position_market()
                    if not closed:
                        # Si no se pudo cerrar, no abrir nueva posición
                        log.error("Fallo en cierre. No se abrirá posición contraria.")
                        await asyncio.sleep(LOOP_SEC)
                        continue

                    # Pequeña pausa para que Binance registre el cierre
                    await asyncio.sleep(0.5)

                # 2. Abrir nueva posición en la dirección de la señal
                await self.open_position_market(signal, ref_price)

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
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

async def main():
    bot = EMABot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        log.info("Apagado manual.")
        tg("🔴 <b>Bot detenido manualmente.</b>")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
