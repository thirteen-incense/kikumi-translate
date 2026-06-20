import os
import re
import time
import requests


class KikumiAPI:
    def __init__(self, base_url: str, worker_name: str, username: str = "admin", password: str = "admin"):
        self.base_url = base_url.rstrip("/")
        self.worker_name = worker_name
        self.username = username
        self.password = password
        self.session = requests.Session()
        self._login_with_retry()

    def _login_with_retry(self, delay: int = 5):
        """登录 kikumi，失败则一直重试（应对 kikumi 还没启动好的情况）"""
        retried = False
        while True:
            if self._login(self.username, self.password):
                if retried:
                    print("已重新连接")
                return
            retried = True
            print(f"认证失败，{delay} 秒后重试...")
            time.sleep(delay)

    def _login(self, username: str, password: str) -> bool:
        """登录 kikumi 以获取 session cookie（/files 等路由需要认证）
        返回 True 表示登录成功"""
        try:
            # 1. 获取登录页，取 CSRF token
            login_resp = self.session.get(f"{self.base_url}/login", timeout=10, allow_redirects=False)
            print(f"GET /login -> HTTP {login_resp.status_code} (user={username}, password={password})", flush=True)
            if login_resp.status_code != 200:
                return False

            # 从 meta 标签取 CSRF token（比 form input 更可靠）
            match = re.search(r'csrf-token" content="([^"]+)"', login_resp.text)
            if not match:
                print("未找到 CSRF token", flush=True)
                return False
            csrf_token = match.group(1)

            # 2. 提交登录表单（必须跟随重定向，否则 session cookie 不会被保存）
            res = self.session.post(
                f"{self.base_url}/session",
                data={
                    "_csrf_token": csrf_token,
                    "username": username,
                    "password": password,
                },
                timeout=10,
            )
            print(f"POST /session -> HTTP {res.status_code}", flush=True)
            # 验证：访问受保护页面，检查是否真正登录成功
            verify = self.session.get(f"{self.base_url}/scan", timeout=5, allow_redirects=False)
            if verify.status_code == 200:
                print(f"认证成功: {username} / {password}")
                return True
            print(f"密码错误 (重定向到 {verify.headers.get('location', '?')})", flush=True)
            return False
        except Exception as e:
            print(f"认证异常: {e}")
            return False

    def register(self) -> bool:
        try:
            res = self.session.post(
                f"{self.base_url}/api/translate/register",
                json={"worker_name": self.worker_name},
                timeout=10,
            )
            return res.json().get("success", False)
        except (requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError, ValueError):
            return False

    def acquire(self) -> dict | None:
        try:
            res = self.session.post(
                f"{self.base_url}/api/translate/acquire",
                json={"worker_name": self.worker_name},
                timeout=10,
            )
            data = res.json()
            if data.get("success") and data.get("task"):
                return data["task"]
        except (requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError, ValueError):
            pass
        return None

    def download_audio(self, audio_url: str, audio_name: str, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, audio_name)
        # 不跟随重定向！避免 session 失效时静默下载到登录页 HTML
        try:
            res = self.session.get(f"{self.base_url}{audio_url}", stream=True, timeout=30, allow_redirects=False)
        except requests.exceptions.ConnectionError as e:
            print(f"音频下载连接失败: {e}，尝试重新登录后重试...", flush=True)
            self._login(self.username, self.password)
            print("已重新连接")
            res = self.session.get(f"{self.base_url}{audio_url}", stream=True, timeout=30, allow_redirects=False)
        if res.status_code == 302:
            # session 失效，重新登录后重试
            print("音频下载被重定向，session 可能已失效，重新登录...", flush=True)
            self._login(self.username, self.password)
            print("已重新连接")
            res = self.session.get(f"{self.base_url}{audio_url}", stream=True, timeout=30, allow_redirects=False)
        res.raise_for_status()
        with open(path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
        return path

    def update_status(self, task_id: int, secret: str, status: str) -> bool:
        try:
            res = self.session.post(
                f"{self.base_url}/api/translate/status",
                json={"id": task_id, "secret": secret, "worker_status": status},
                timeout=10,
            )
            return res.json().get("success", False)
        except (requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError, ValueError):
            return False

    def heartbeat(self) -> bool:
        try:
            return self.register()
        except requests.exceptions.ConnectionError:
            return False

    def finish(self, task_id: int, secret: str, lrc_content: str) -> bool:
        try:
            res = self.session.post(
                f"{self.base_url}/api/translate/finish",
                json={"id": task_id, "secret": secret, "lrc_content": lrc_content},
                timeout=30,
            )
            return res.json().get("success", False)
        except (requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError, ValueError):
            return False
