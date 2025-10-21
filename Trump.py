# -*- coding: utf-8 -*-
import configparser
import json
import logging
import time
import requests
import feedparser
import os
import re
import datetime
import hashlib
from typing import Any, Dict, Optional, List
from zoneinfo import ZoneInfo  # Python 3.9+
import email.utils
import calendar
from dataclasses import dataclass
import html
import unicodedata
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# 尝试使用 BeautifulSoup 提升 HTML 清洗质量；没有则回退到正则
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

# =========================
# 日志初始化（避免重复 handlers）
# =========================
def init_logging(log_file: str, log_level: str = "INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

# =========================
# 工具：读取多行字段
# =========================
def _cfg_get_multiline(cfg: configparser.ConfigParser, section: str, option: str, fallback: str = "") -> str:
    if cfg.has_option(section, option):
        return cfg.get(section, option).strip()
    return fallback.strip()

# =========================
# 文本规范化 / HTML 清洗 / URL 标准化（新增）
# =========================
def _normalize_unicode(s: str) -> str:
    """Unicode 规范化，降低标点/空格变体导致的误差"""
    return unicodedata.normalize("NFKC", s or "")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def _strip_html_keep_text(s: str) -> str:
    """
    将 HTML → 纯文本：
    1) html.unescape 反转义实体
    2) 首选 BeautifulSoup 去标签；若不可用退化到正则
    3) 合并多余空白
    4) NFKC 规范化
    """
    if not s:
        return ""
    s = html.unescape(s)
    if _HAS_BS4:
        try:
            text = BeautifulSoup(s, "html.parser").get_text(separator=" ", strip=True)
        except Exception:
            text = _TAG_RE.sub(" ", s)
    else:
        text = _TAG_RE.sub(" ", s)
    text = _WS_RE.sub(" ", text).strip()
    return _normalize_unicode(text)

def _best_entry_body(entry) -> str:
    """
    选择正文的“最佳可用字段”：
    content[].value → summary → description → ""
    并做 HTML → 纯文本清洗
    """
    # content
    if hasattr(entry, "content"):
        try:
            contents = entry.content or []
            for c in contents:
                v = c.get("value") if isinstance(c, dict) else getattr(c, "value", None)
                v = _strip_html_keep_text(v)
                if v:
                    return v
        except Exception:
            pass
    # summary
    if hasattr(entry, "summary"):
        v = _strip_html_keep_text(getattr(entry, "summary", "") or "")
        if v:
            return v
    # description
    if hasattr(entry, "description"):
        v = _strip_html_keep_text(getattr(entry, "description", "") or "")
        if v:
            return v
    return ""

def _normalize_link(url: Optional[str]) -> Optional[str]:
    """
    URL 标准化：
    - 去掉 #fragment
    - 过滤 utm_* 等追踪参数
    - 统一域名小写
    - 去掉尾随斜杠（保留根 '/'）
    """
    if not url:
        return None
    try:
        p = urlparse(url)
        fragment = ""
        q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            if k.lower().startswith("utm_"):
                continue
            q.append((k, v))
        query = urlencode(q, doseq=True)
        path = p.path.rstrip("/") or "/"
        normalized = urlunparse((p.scheme, p.netloc.lower(), path, p.params, query, fragment))
        return normalized
    except Exception:
        return url

_WEAK_TITLE_RE = re.compile(r"^\[?No Title\]? - Post from", re.IGNORECASE)

def entry_text_for_llm(entry) -> str:
    """
    构造送模型的“净文本”：标题（若非系统占位） + 正文
    """
    title_raw = getattr(entry, "title", "") or ""
    title = _strip_html_keep_text(title_raw)
    weak_title = bool(_WEAK_TITLE_RE.match(title))
    body = _best_entry_body(entry)

    if title and not weak_title:
        if body:
            return f"{title}\n{body}"
        return title
    return body

def is_effectively_empty(text: str, min_len: int = 8) -> bool:
    """清洗后的可见文本长度是否过短（默认 <8 视为空）"""
    if not text:
        return True
    return len(text.strip()) < min_len

def stable_fingerprint(entry) -> str:
    """
    稳定指纹：优先 id/guid/link（标准化 URL）→ 回退 title+pubDate → 最后 str(entry)
    对原料先做 Unicode 规范化，URL 标准化，降低误判/漏判。
    """
    parts: List[str] = []

    cand = None
    for k in ("id", "guid", "link"):
        if hasattr(entry, k) and getattr(entry, k):
            v = str(getattr(entry, k))
            if k == "link":
                v = _normalize_link(v) or v
            cand = v
            break
    if cand:
        parts.append(_normalize_unicode(cand))
    else:
        title = _normalize_unicode(getattr(entry, "title", "") or "")
        pub = ""
        for k in ("published", "pubDate", "updated", "created", "dc_date"):
            if hasattr(entry, k) and getattr(entry, k):
                pub = _normalize_unicode(str(getattr(entry, k)))
                break
        if title:
            parts.append(title)
        if pub:
            parts.append(pub)
        if not parts:
            parts.append(_normalize_unicode(str(entry)))

    raw = "||".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

# =========================
# 推送抽象与实现
# =========================
class Pusher:
    def push(self, message: str, title: Optional[str] = None, tags: Optional[str] = None):
        raise NotImplementedError

class ServerChanSDKPusher(Pusher):
    def __init__(self, send_key: str, session: requests.Session,
                 default_title: str = "特朗普推文预警",
                 default_tags: str = "服务器报警|图片",
                 max_title_len: int = 64,
                 max_desp_len: int = 512,
                 uid_override: Optional[str] = None):
        self.send_key = send_key
        self.session = session
        self.default_title = default_title
        self.default_tags = default_tags
        self.MAX_TITLE_LEN = max_title_len
        self.MAX_DESP_LEN = max_desp_len

        if uid_override:
            self.uid = uid_override
        else:
            m = re.search(r"sctp(\d+)t", send_key or "")
            self.uid = m.group(1) if m else None
            if not self.uid:
                logging.error("send_key 格式不匹配且未提供 serverchan.uid，无法推送")

    def push(self, message: str, title: Optional[str] = None, tags: Optional[str] = None):
        if not self.uid:
            logging.error("未能提取 uid 无法推送")
            return
        safe_title = (title or self.default_title)[: self.MAX_TITLE_LEN]
        safe_desp = (message or "")[: self.MAX_DESP_LEN]
        safe_tags = tags or self.default_tags

        url = f"https://{self.uid}.push.ft07.com/send/{self.send_key}.send"
        payload = {"title": safe_title, "desp": safe_desp, "tags": safe_tags}
        headers = {"Content-Type": "application/json"}
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=10)
            print(resp.text)  # 原始返回便于排错
            if resp.status_code != 200:
                logging.error(f"ServerChan API 推送失败: {resp.text}")
        except Exception as e:
            logging.error(f"ServerChan API 推送失败: {e}")

class PusherFactory:
    @staticmethod
    def create_pusher(method: str, send_key: str, session: requests.Session,
                      title: str, tags: str, max_title_len: int, max_desp_len: int, uid_override: Optional[str]):
        if method == "serverchan_sdk" and send_key:
            return ServerChanSDKPusher(send_key, session, title, tags, max_title_len, max_desp_len, uid_override)
        else:
            logging.error("当前仅支持 serverchan_sdk 推送方式")
            return None

# =========================
# 稳健 JSON 解析与校验
# =========================
ALLOWED_IMPORTANCE = {"high", "medium", "low", "none"}
IMPORTANCE_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

def validate_result(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    impact = obj.get("impact", None)
    if not isinstance(impact, bool):
        return None
    importance = str(obj.get("importance", "none")).lower().strip()
    if importance not in ALLOWED_IMPORTANCE:
        return None
    asset_class = obj.get("asset_class", [])
    if asset_class is None:
        asset_class = []
    if not isinstance(asset_class, list) or not all(isinstance(x, str) for x in asset_class):
        return None
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        return None
    original_cn = obj.get("original_cn", "")
    if not isinstance(original_cn, str):
        return None
    return {
        "impact": impact,
        "importance": importance,
        "asset_class": asset_class,
        "reason": reason.strip(),
        "original_cn": original_cn.strip()
    }

def iter_possible_json_blobs(text: str) -> List[str]:
    candidates: List[str] = []

    fence_patterns = [
        (r"```json\s*(.*?)\s*```", re.DOTALL),
        (r"```\s*(\{[\s\S]*?\})\s*```", re.DOTALL)
    ]
    for pat, flag in fence_patterns:
        for m in re.finditer(pat, text, flags=flag):
            candidates.append(m.group(1))

    s = text
    opens = [i for i, ch in enumerate(s) if ch == '{']
    if len(opens) <= 300:
        for i in opens:
            depth = 0
            for j in range(i, len(s)):
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                    if depth == 0:
                        if j + 1 - i <= 10000:
                            candidates.append(s[i:j+1])
                        break

    m = re.search(r"\{[\s\S]*\}", text)
    if m and len(m.group(0)) <= 10000:
        candidates.append(m.group(0))

    seen = set()
    uniq: List[str] = []
    for c in candidates:
        key = hashlib.sha256(c.encode('utf-8', errors='ignore')).hexdigest()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def robust_extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        valid = validate_result(obj)
        if valid:
            return valid
    except Exception:
        pass

    for cand in iter_possible_json_blobs(text):
        try:
            obj = json.loads(cand)
            valid = validate_result(obj)
            if valid:
                return valid
        except Exception:
            continue
    return None

# =========================
# RSS 条目时间解析
# =========================
def parse_datetime_any(dt_str: str) -> Optional[datetime.datetime]:
    if not dt_str:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(dt_str)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        pass
    try:
        s = dt_str.strip().replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        return None

def get_entry_datetime(entry) -> Optional[datetime.datetime]:
    for k in ("published_parsed", "updated_parsed", "created_parsed"):
        if hasattr(entry, k):
            st = getattr(entry, k)
            if st:
                try:
                    ts = calendar.timegm(st)
                    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                except Exception:
                    pass
    for k in ("published", "pubDate", "updated", "created", "dc_date"):
        if hasattr(entry, k):
            dt = parse_datetime_any(getattr(entry, k))
            if dt:
                return dt.astimezone(datetime.timezone.utc)
    return None

def parse_rfc822_dt(dt_str: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.strptime(dt_str, "%a, %d %b %Y %H:%M:%S %z")
    except Exception:
        try:
            return email.utils.parsedate_to_datetime(dt_str)
        except Exception:
            return None

# =========================
# 轻量 POST 重试包装
# =========================
@dataclass
class RetryConfig:
    retries: int = 2
    backoff: float = 0.8  # seconds

def post_with_retry(session: requests.Session, url: str, *, headers=None, params=None, json_body=None,
                    timeout: float = 30.0, retry: RetryConfig = RetryConfig()):
    last_exc = None
    for i in range(retry.retries + 1):
        try:
            resp = session.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise requests.HTTPError(f"status={resp.status_code}, body={resp.text[:200]}")
            return resp
        except Exception as e:
            last_exc = e
            if i < retry.retries:
                time.sleep(retry.backoff * (i + 1))
            else:
                break
    raise last_exc

# =========================
# 主监控器
# =========================
class TrumpTweetMonitor:
    def __init__(self, config_path="config.ini"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        self.config = configparser.ConfigParser()
        with open(config_path, "r", encoding="utf-8") as f:
            self.config.read_file(f)

        self.log_file = self.config.get("settings", "log_file", fallback="tweet_monitor.log")
        self.log_level = self.config.get("settings", "log_level", fallback="INFO")
        init_logging(self.log_file, self.log_level)

        self.rss_url = self.config.get("feed", "rss_url")
        self.model_provider = self.config.get("model", "provider", fallback="openai").lower()
        self.model_name = self.config.get("model", "model_name", fallback="gpt-4o")
        self.openai_key = self.config.get("openai", "api_key", fallback=None)
        self.openai_api_base = self.config.get("openai", "api_base", fallback="https://api.openai.com/v1")
        self.gemini_key = self.config.get("gemini", "api_key", fallback=None)

        self.check_interval = self.config.getint("settings", "check_interval", fallback=60)
        self.proxy_enable = self.config.getboolean("settings", "proxy_enable", fallback=True)
        self.proxy_http = self.config.get("settings", "proxy_http", fallback="http://127.0.0.1:7890")
        self.proxy_https = self.config.get("settings", "proxy_https", fallback="http://127.0.0.1:7890")
        self.analysis_record_file = self.config.get("settings", "analysis_record_file", fallback="analysis_records.json")
        self.analyzed_guids_file = self.config.get("settings", "analyzed_guids_file", fallback="analyzed_guids.json")
        self.analyzed_guids = self.load_analyzed_guids()

        # ---- 时区 & start_time 正确处理（默认 Asia/Singapore）----
        self.timezone_name = self.config.get("settings", "timezone", fallback="Asia/Singapore")
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            logging.warning(f"未知时区 {self.timezone_name}，回退使用 UTC")
            self.tz = datetime.timezone.utc

        start_time_str = self.config.get("settings", "start_time", fallback=None)
        if start_time_str:
            try:
                naive = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
                localized = naive.replace(tzinfo=self.tz)
                self.start_time_utc = localized.astimezone(datetime.timezone.utc)
            except Exception as e:
                logging.error(f"解析 start_time 失败: {e}")
                self.start_time_utc = None
        else:
            self.start_time_utc = None

        self.daily_report_time = self.config.get("settings", "daily_report_time", fallback="08:00")
        self.daily_state_file = self.config.get("settings", "daily_state_file", fallback="daily_report_state.json")

        self.session = requests.Session()
        if self.proxy_enable:
            self.session.proxies.update({"http": self.proxy_http, "https": self.proxy_https})

        self.last_entry_id = None
        self.push_method = self.config.get("server", "push_method", fallback="serverchan_sdk")
        self.send_key = self.config.get("server", "send_key", fallback=None)

        self.prompt_analysis_template = _cfg_get_multiline(self.config, "prompt.analysis", "template")
        self.prompt_analysis_system_role = _cfg_get_multiline(self.config, "prompt.analysis", "system_role")
        self.prompt_analysis_json_schema = _cfg_get_multiline(self.config, "prompt.analysis", "json_schema")
        self.prompt_analysis_rules = _cfg_get_multiline(self.config, "prompt.analysis", "rules")

        self.prompt_translate_template = _cfg_get_multiline(self.config, "prompt.translate", "template")
        self.prompt_summary_template = _cfg_get_multiline(self.config, "prompt.summary", "template")

        self.push_on_impact = self.config.getboolean("push.policy", "push_on_impact", fallback=True)
        self.push_min_importance = self.config.get("push.policy", "push_min_importance", fallback="high").strip().lower()
        self.push_title = self.config.get("push.format", "title", fallback="特朗普推文预警")
        self.push_tags = self.config.get("push.format", "tags", fallback="服务器报警|图片")
        self.push_alert_template = _cfg_get_multiline(
            self.config, "push.format", "alert_template",
            "🚨 高重要性推文：{original_cn}\n影响资产：{asset_str}\n理由：{reason}\n{link_line}"
        )
        self.push_fallback_template = _cfg_get_multiline(
            self.config, "push.format", "fallback_template",
            "⚠️ 未能解析结构化结果，供人工复核：\n摘要/译文：{brief_cn}\n{link_line}"
        )
        self.push_max_title_len = self.config.getint("push.format", "max_title_len", fallback=64)
        self.push_max_desp_len = self.config.getint("push.format", "max_desp_len", fallback=512)

        self.serverchan_uid = self.config.get("serverchan", "uid", fallback="").strip()

        self.model_rpm = self.config.getint("rate_limit", "model_rpm", fallback=30)
        self.push_rpm = self.config.getint("rate_limit", "push_rpm", fallback=20)
        self.max_entries_per_cycle = self.config.getint("rate_limit", "max_entries_per_cycle", fallback=50)
        self.min_sleep_between_entries_ms = self.config.getint("rate_limit", "min_sleep_between_entries_ms", fallback=200)

        now_mono = time.monotonic()
        self._rate_limits = {
            "model": {
                "capacity": self.model_rpm,
                "tokens": float(self.model_rpm),
                "rate_per_sec": self.model_rpm / 60.0,
                "updated": now_mono,
            },
            "push": {
                "capacity": self.push_rpm,
                "tokens": float(self.push_rpm),
                "rate_per_sec": self.push_rpm / 60.0,
                "updated": now_mono,
            },
        }

        self.pusher = PusherFactory.create_pusher(
            self.push_method, self.send_key, self.session,
            self.push_title, self.push_tags, self.push_max_title_len, self.push_max_desp_len,
            self.serverchan_uid or None
        )
        if self.pusher is None:
            raise RuntimeError("推送方式初始化失败")

        logging.info("✅ 初始化完成")

    # ---------- Prompt ----------
    def build_prompt(self, tweet_text: str) -> str:
        if self.prompt_analysis_template:
            return self.prompt_analysis_template.format(
                system_role=self.prompt_analysis_system_role,
                json_schema=self.prompt_analysis_json_schema,
                rules=self.prompt_analysis_rules,
                tweet_text=tweet_text
            ).strip()
        default_template = f"""{self.prompt_analysis_system_role or "你是一名资深全球宏观与多资产策略师。请对下面的特朗普相关文本进行**市场影响**评估。请注意现在的时间节点特朗普已经上台且两院都在共和党手中。"}

务必严格按以下**唯一 JSON**结构输出（不要解释、不要代码块、不要多余文字）：

{self.prompt_analysis_json_schema or '{ "impact": true|false, "importance": "high"|"medium"|"low"|"none", "asset_class": ["US equities","Treasuries","USD","Gold","Oil","China A-shares","EM FX"], "reason": "用中文简明解释为什么（最多200字）", "original_cn": "对原文的准确中文翻译或提炼（最多200字）" }'}

**评估要点（非常重要）：**
{self.prompt_analysis_rules or ""}

需要评估的原文如下：
{tweet_text}"""
        return default_template.strip()

    # ---------- 令牌桶限速 ----------
    def _acquire_token(self, bucket: str, permits: int = 1, timeout_sec: float = 10.0) -> bool:
        if bucket not in self._rate_limits:
            return True
        rl = self._rate_limits[bucket]
        deadline = time.monotonic() + timeout_sec
        while True:
            now = time.monotonic()
            elapsed = max(0.0, now - rl["updated"])
            refill = elapsed * rl["rate_per_sec"]
            if refill > 0:
                rl["tokens"] = min(rl["capacity"], rl["tokens"] + refill)
                rl["updated"] = now
            if rl["tokens"] >= permits:
                rl["tokens"] -= permits
                return True
            need = permits - rl["tokens"]
            wait_sec = need / rl["rate_per_sec"] if rl["rate_per_sec"] > 0 else 0.5
            if wait_sec <= 0:
                wait_sec = 0.05
            if time.monotonic() + wait_sec > deadline:
                return False
            time.sleep(min(wait_sec, 1.0))

    def push_with_rate_limit(self, message: str):
        if not self._acquire_token("push", permits=1, timeout_sec=15.0):
            logging.warning("推送限速未获得令牌，本次跳过")
            return
        self.pusher.push(message, title=self.push_title, tags=self.push_tags)

    # ---------- 稳健模型调用 ----------
    def analyze_with_model(self, tweet_text):
        prompt = self.build_prompt(tweet_text)
        if self.model_provider == "openai":
            return self.analyze_with_openai(prompt)
        elif self.model_provider == "gemini":
            return self.analyze_with_gemini(prompt)
        else:
            logging.error(f"不支持的模型提供商: {self.model_provider}")
            return None

    def analyze_with_openai(self, prompt):
        try:
            if not self._acquire_token("model", permits=1, timeout_sec=15.0):
                logging.warning("模型限速未获得令牌，本次跳过 openai 调用")
                return None
            url = f"{self.openai_api_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = post_with_retry(self.session, url, headers=headers, json_body=data, timeout=30.0)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return robust_extract_json(content)
        except Exception as e:
            logging.error(f"大模型 处理出错: {e}")
            return None

    def analyze_with_gemini(self, prompt):
        try:
            if not self._acquire_token("model", permits=1, timeout_sec=15.0):
                logging.warning("模型限速未获得令牌，本次跳过 gemini 调用")
                return None
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            headers = {"Content-Type": "application/json"}
            params = {"key": self.gemini_key}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            response = post_with_retry(self.session, url, headers=headers, params=params, json_body=data, timeout=30.0)
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return robust_extract_json(content)
        except Exception as e:
            logging.error(f"Gemini 处理出错: {e}")
            return None

    # ---------- 兜底功能 ----------
    def quick_translate_cn(self, text: str) -> Optional[str]:
        tmpl = self.prompt_translate_template or "请把以下英文或混合文本准确翻译成简洁中文（最多120字），只输出译文：\n{source_text}"
        prompt = tmpl.format(source_text=text)
        try:
            if self.model_provider == "openai":
                if not self._acquire_token("model", permits=1, timeout_sec=15.0):
                    logging.warning("模型限速未获得令牌，跳过 quick_translate_cn")
                    return None
                url = f"{self.openai_api_base}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}]}
                resp = post_with_retry(self.session, url, headers=headers, json_body=data, timeout=30.0)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif self.model_provider == "gemini":
                if not self._acquire_token("model", permits=1, timeout_sec=15.0):
                    logging.warning("模型限速未获得令牌，跳过 quick_translate_cn")
                    return None
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.gemini_key}
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = post_with_retry(self.session, url, headers=headers, params=params, json_body=data, timeout=30.0)
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logging.error(f"快速翻译失败: {e}")
        return None

    def quick_summarize(self, text: str) -> Optional[str]:
        tmpl = self.prompt_summary_template or "请将以下内容总结为一份简洁的中文日报（200字内，突出高重要性与市场影响）：\n{source_text}"
        prompt = tmpl.format(source_text=text)
        try:
            if self.model_provider == "openai":
                if not self._acquire_token("model", permits=1, timeout_sec=15.0):
                    logging.warning("模型限速未获得令牌，跳过 quick_summarize")
                    return None
                url = f"{self.openai_api_base}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}]}
                resp = post_with_retry(self.session, url, headers=headers, json_body=data, timeout=30.0)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif self.model_provider == "gemini":
                if not self._acquire_token("model", permits=1, timeout_sec=15.0):
                    logging.warning("模型限速未获得令牌，跳过 quick_summarize")
                    return None
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.gemini_key}
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = post_with_retry(self.session, url, headers=headers, params=params, json_body=data, timeout=30.0)
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logging.error(f"日报摘要失败: {e}")
        return None

    # ---------- 记录/读写 ----------
    def save_analysis_record(self, tweet_text, result):
        record = {
            "timestamp": datetime.datetime.now(self.tz).isoformat(),
            "tweet_text": tweet_text,
            "result": result
        }
        try:
            records = []
            if os.path.exists(self.analysis_record_file):
                with open(self.analysis_record_file, "r", encoding="utf-8") as f:
                    try:
                        records = json.load(f)
                    except Exception:
                        records = []
            records.append(record)
            with open(self.analysis_record_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存分析记录失败: {e}")

    def summarize_daily_report(self):
        today_local = datetime.datetime.now(self.tz).date()
        yesterday_local = today_local - datetime.timedelta(days=1)
        summary_records = []
        try:
            if not os.path.exists(self.analysis_record_file):
                return None
            with open(self.analysis_record_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                ts = datetime.datetime.fromisoformat(rec["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=self.tz)
                ts_local = ts.astimezone(self.tz)
                if ts_local.date() == yesterday_local:
                    summary_records.append(rec)
        except Exception as e:
            logging.error(f"读取分析记录失败: {e}")
            return None
        if not summary_records:
            return None
        summary_text = "昨日特朗普推文市场影响日报：\n"
        for rec in summary_records:
            result = rec["result"]
            if not result:
                raw = (rec.get('tweet_text', '') or '').replace("\n", " ")
                summary_text += f"- （未解析）{raw[:80]}...\n"
                continue
            summary_text += (
                f"- {result.get('original_cn','')} | 影响: {result.get('impact')} | "
                f"重要性: {result.get('importance')} | 资产: {result.get('asset_class')} | "
                f"理由: {result.get('reason')}\n"
            )
        summary = self.quick_summarize(summary_text)
        return summary or summary_text

    def send_daily_report(self):
        summary = self.summarize_daily_report()
        if summary:
            self.push_with_rate_limit(f"【特朗普推文市场影响日报】\n{summary}")
        else:
            logging.info("无昨日分析记录，无需推送日报。")

    def load_analyzed_guids(self):
        if os.path.exists(self.analyzed_guids_file):
            try:
                with open(self.analyzed_guids_file, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def save_analyzed_guids(self):
        try:
            with open(self.analyzed_guids_file, "w", encoding="utf-8") as f:
                json.dump(list(self.analyzed_guids), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存已分析指纹失败: {e}")

    def _load_daily_state(self) -> Dict[str, Any]:
        if os.path.exists(self.daily_state_file):
            try:
                with open(self.daily_state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_sent_date": ""}

    def _save_daily_state(self, state: Dict[str, Any]):
        try:
            with open(self.daily_state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存日报状态失败: {e}")

    # ---------- 主循环 ----------
    def run(self):
        while True:
            try:
                logging.info("🚀 进入主循环")

                # ---- 日报触发（窗口判定）----
                try:
                    state = self._load_daily_state()
                    now_local = datetime.datetime.now(self.tz)
                    today_key = now_local.strftime("%Y-%m-%d")
                    try:
                        hh, mm = map(int, (self.daily_report_time or "08:00").split(":"))
                        target = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    except Exception:
                        target = now_local.replace(hour=8, minute=0, second=0, microsecond=0)
                    window_sec = 60
                    diff = (now_local - target).total_seconds()
                    logging.debug(f"⏰ 日报检查：now={now_local}, target={target}, diff={diff:.1f}s, last_sent={state.get('last_sent_date')}")
                    if state.get("last_sent_date") != today_key and 0 <= diff < window_sec:
                        logging.info("📨 触发日报发送（窗口命中）")
                        self.send_daily_report()
                        state["last_sent_date"] = today_key
                        self._save_daily_state(state)
                except Exception as e:
                    logging.error(f"日报触发错误: {e}")

                # ---- 拉取RSS ----
                logging.info("🔄 开始一次 RSS 轮询")
                try:
                    resp = self.session.get(self.rss_url, timeout=15)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.content)
                except Exception as e:
                    logging.error(f"RSS 获取失败: {e}")
                    time.sleep(self.check_interval)
                    continue

                count = len(getattr(feed, "entries", []) or [])
                logging.info(f"📥 本次轮询获取到条目数: {count}")
                if not count:
                    logging.warning("⚠️ 没有获取到推文")
                    time.sleep(self.check_interval)
                    continue

                entries = list(feed.entries)[: self.max_entries_per_cycle]
                logging.info(f"🧮 本轮计划处理条目数: {len(entries)}（上限 {self.max_entries_per_cycle}）")
                for entry in entries:
                    pub_dt_utc = get_entry_datetime(entry)
                    if pub_dt_utc:
                        logging.debug(f"条目时间: {pub_dt_utc.isoformat()} vs 起始阈值: {self.start_time_utc}")
                    else:
                        logging.debug("条目时间: <未解析> vs 起始阈值: %s", self.start_time_utc)

                    if not pub_dt_utc:
                        logging.debug("⏭️ 跳过：无法解析发布时间（无 *_parsed / 无法解析字符串）")
                        continue

                    if self.start_time_utc and pub_dt_utc < self.start_time_utc:
                        logging.debug(f"⏭️ 跳过：早于 start_time（{pub_dt_utc.isoformat()} < {self.start_time_utc.isoformat()}）")
                        continue

                    # —— 使用“稳定”指纹去重（含 URL/Unicode 标准化）——
                    fp = stable_fingerprint(entry)
                    if fp in self.analyzed_guids:
                        logging.debug("⏭️ 跳过：指纹已存在（去重命中）")
                        continue

                    # —— 构造净文本并过滤空内容 —— 
                    tweet_text = entry_text_for_llm(entry)
                    if is_effectively_empty(tweet_text):
                        logging.debug("⏭️ 跳过：净文本为空（标题/正文均无有效内容）")
                        # 即便空，也记 fingerprint，防止源重复推空帖反复处理
                        self.analyzed_guids.add(fp)
                        self.save_analyzed_guids()
                        continue

                    link = _normalize_link(getattr(entry, "link", None))
                    logging.info(f"🆕 检测到新推文: {(_strip_html_keep_text(getattr(entry, 'title', '') or '') or '[无标题]')}")
                    result = self.analyze_with_model(tweet_text)

                    if result:
                        logging.info(f"✅ 分析结果: {result}")
                        self.save_analysis_record(tweet_text, result)

                        imp = str(result.get("importance", "none")).lower().strip()
                        impact_flag = bool(result.get("impact") is True)
                        threshold = IMPORTANCE_ORDER.get(self.push_min_importance, 3)
                        rank = IMPORTANCE_ORDER.get(imp, 0)
                        should_push = (not self.push_on_impact or impact_flag) and (rank >= threshold)

                        if should_push:
                            asset = result.get("asset_class")
                            asset_str = ", ".join(asset) if isinstance(asset, list) else str(asset or "")
                            link_line = f"原文链接：{link}" if link else ""
                            msg = self.push_alert_template.format(
                                original_cn=result.get("original_cn", ""),
                                asset_str=asset_str,
                                reason=result.get("reason", ""),
                                link_line=link_line
                            )
                            self.push_with_rate_limit(msg)
                    else:
                        logging.warning("⚠️ 模型未返回有效 JSON，执行回退推送")
                        brief_cn = self.quick_translate_cn(tweet_text) or "(翻译失败)"
                        link_line = f"原文链接：{link}" if link else ""
                        fallback_msg = self.push_fallback_template.format(brief_cn=brief_cn, link_line=link_line)
                        self.push_with_rate_limit(fallback_msg)
                        self.save_analysis_record(tweet_text, None)

                    self.analyzed_guids.add(fp)
                    self.save_analyzed_guids()

                    if self.min_sleep_between_entries_ms > 0:
                        time.sleep(self.min_sleep_between_entries_ms / 1000.0)

                logging.info("✅ 本轮 RSS 处理结束，进入休眠")
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"运行时错误: {e}")
                time.sleep(self.check_interval)

# =========================
# 入口
# =========================
if __name__ == '__main__':
    monitor = TrumpTweetMonitor()
    monitor.run()
