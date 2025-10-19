---

# Trump Tweet Monitor

Monitors Donald Trump’s RSS feed, uses an LLM (OpenAI or Gemini) to evaluate market impact, and pushes alerts via Server酱.

---

## Quick Start

```bash
git clone https://github.com/yourname/trump-tweet-monitor.git
cd trump-tweet-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install requests feedparser
cp config.example.ini config.ini
python trump_tweet_monitor.py
```

---

## Configuration (`config.ini`)

Below is what each section means.

### `[settings]`

| Key                         | Description                                     |
| --------------------------- | ----------------------------------------------- |
| `log_file`                  | Path to log file (default: `tweet_monitor.log`) |
| `log_level`                 | Logging level: `INFO`, `DEBUG`, etc.            |
| `check_interval`            | Seconds between RSS checks                      |
| `timezone`                  | Local timezone (e.g., `Asia/Taipei`)            |
| `daily_report_time`         | Local time (HH:MM) to send daily summary        |
| `analysis_record_file`      | File storing all analysis results               |
| `analyzed_guids_file`       | File storing processed tweet fingerprints       |
| `proxy_enable`              | Whether to use proxy (`true` / `false`)         |
| `proxy_http`, `proxy_https` | Proxy URLs (if enabled)                         |

---

### `[feed]`

| Key       | Description                                          |
| --------- | ---------------------------------------------------- |
| `rss_url` | RSS feed to monitor (default: Trump feed via nitter) |

---

### `[model]`

| Key          | Description                                  |
| ------------ | -------------------------------------------- |
| `provider`   | `openai` or `gemini`                         |
| `model_name` | Model name (e.g. `gpt-4o`, `gemini-1.5-pro`) |

---

### `[openai]` / `[gemini]`

| Key        | Description                             |
| ---------- | --------------------------------------- |
| `api_key`  | Your model API key                      |
| `api_base` | (OpenAI only) custom endpoint if needed |

---

### `[server]`

| Key           | Description      |
| ------------- | ---------------- |
| `push_method` | `serverchan_sdk` |
| `send_key`    | Server酱 SDK key  |

---

### `[serverchan]`

| Key   | Description                                                |
| ----- | ---------------------------------------------------------- |
| `uid` | Optional UID; leave blank if `send_key` already encodes it |

---

### `[rate_limit]`

| Key                            | Description                        |
| ------------------------------ | ---------------------------------- |
| `model_rpm`                    | Max model API calls per minute     |
| `push_rpm`                     | Max push requests per minute       |
| `max_entries_per_cycle`        | Max RSS entries per loop           |
| `min_sleep_between_entries_ms` | Milliseconds delay between entries |

---

### `[prompt.analysis]`

Defines the main LLM prompt.
You can customize:

* `system_role` – model’s role description
* `json_schema` – required JSON structure
* `rules` – short bullet rules
* `template` – how the final prompt is composed

---

### `[prompt.translate]`

Short fallback translation prompt for when JSON fails.

### `[prompt.summary]`

Prompt for daily summary message.

---

### `[push.policy]`

| Key                   | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `push_on_impact`      | Only push when `impact=true`                                |
| `push_min_importance` | Minimum importance to trigger push (`none/low/medium/high`) |

---

### `[push.format]`

| Key                             | Description                      |
| ------------------------------- | -------------------------------- |
| `title`, `tags`                 | Push title and tag labels        |
| `alert_template`                | Template for successful analysis |
| `fallback_template`             | Template when JSON parsing fails |
| `max_title_len`, `max_desp_len` | Truncation limits                |

---

## Run as Service (optional)

```bash
sudo systemctl enable trump-monitor
sudo systemctl start trump-monitor
```

---

## Output Files

* `tweet_monitor.log` – logs
* `analysis_records.json` – all analyses
* `analyzed_guids.json` – processed tweet IDs

---

## License

MIT License

