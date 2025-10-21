# Trump Tweet Monitor

监控唐纳德·特朗普的 RSS feed，使用大语言模型（OpenAI 或 Gemini）评估其市场影响，并通过 Server酱 推送警报。

## 目录

  - [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
    - [安装步骤](#安装步骤)
    - [配置说明](#配置说明)
    - [使用说明](#使用说明)
    - [部署](#部署)
    - [使用到的框架与服务](#使用到的框架与服务)
    - [作者](#作者)



### 上手指南

本服务建议运行在服务器上，可以使用Render、Fly.io、以及阿里云等方式部署。


###### **开发前的配置要求**

  - Python 3.x
  - pip

###### **安装步骤**

1.  克隆本项目
    ```sh
    git clone https://github.com/EliotYang/trump-tweet-monitor.git
    ```
2.  进入项目目录并创建虚拟环境
    ```bash
    cd trump-tweet-monitor
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  安装依赖
    ```sh
    pip install -U pip
    pip install requests feedparser
    ```
4.  配置 API 密钥及其他设置
    ```sh
    cp config.example.ini config.ini
    ```
    *然后编辑 `config.ini` 文件，填入你的 OpenAI/Gemini API 密钥和 Server酱 SendKey。*

### 配置说明

项目的所有配置都在 `config.ini` 文件中，详细说明请参照内部的中文翻译。

### 使用说明

直接运行主脚本即可启动监控：

```bash
python trump_tweet_monitor.py
```

程序会根据 `config.ini` 中 `check_interval` 的设定，周期性地检查 RSS feed，分析新条目，并根据推送策略发送通知。

**输出文件:**

  * `tweet_monitor.log` – 程序运行日志
  * `analysis_records.json` – 所有的分析结果记录
  * `analyzed_guids.json` – 已处理过的推文 ID，用于去重

### 部署

你可以将此脚本作为 systemd 服务在后台持续运行。

```bash
# 以下为示例服务配置，请根据实际路径修改
# /etc/systemd/system/trump-monitor.service

[Unit]
Description=Trump Tweet Monitor Service
After=network.target

[Service]
User=your_user # 替换为你的用户名
WorkingDirectory=/path/to/trump-tweet-monitor # 替换为项目路径
ExecStart=/path/to/trump-tweet-monitor/.venv/bin/python trump_tweet_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
```

然后启用并启动服务：

```bash
sudo systemctl enable trump-monitor
sudo systemctl start trump-monitor
```

### 使用到的框架与服务

  - [Python](https://www.python.org/)
  - [OpenAI API](https://openai.com/blog/openai-api) / [Google Gemini API](https://ai.google.dev/)
  - [Server酱](https://www.google.com/search?q=http://sc.ftqq.com/)

### 作者

**EliotYang**  [EliotYang](https://www.google.com/search?q=https://github.com/EliotYang)

小红书账号： [城下](https://www.xiaohongshu.com/user/profile/680df0be000000000e012b52)
