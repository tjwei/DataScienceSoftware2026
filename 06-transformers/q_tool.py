from datetime import datetime

def get_time() -> str:
    """
    Get the current date and time (including the day of the week).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

tools = [get_weather, calculator, get_time]
print("Phrasing 1:", run_agent("What day of the week is it today?"))
print("Phrasing 2:", run_agent("Please use a tool to look up the current time, then tell me what day of the week it is today."))
