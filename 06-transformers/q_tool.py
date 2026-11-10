from datetime import datetime

def get_time() -> str:
    """
    取得現在的日期與時間 (包含星期幾)。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

tools = [get_weather, calculator, get_time]
print("問法一:", run_agent("今天星期幾？"))
print("問法二:", run_agent("請使用工具查詢現在的時間，然後告訴我今天星期幾。"))
