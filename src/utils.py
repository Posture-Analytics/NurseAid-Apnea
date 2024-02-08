from datetime import datetime

def format_time(timestamp: str) -> str:
    """ Generate a formatted string from a millisecond timestamp

    Args:
        timestamp (str): A millisecond timestamp

    Returns:
        str: A formatted string with the timestamp
    """

    # Convert the string of a millisecond timestamp into a int of a second timestamp
    timestamp = int(timestamp) / 1000

    # Convert the timestamp into a datetime object
    timestamp = datetime.fromtimestamp(timestamp)

    # Convert the datetime object into a formatted string
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Chop off the last 3 digits of the string (the microseconds) and add the timezone
    return time_str[:-3] + " GMT-3"