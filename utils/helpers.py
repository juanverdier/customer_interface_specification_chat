import datetime
import random
import string
import threading
import langfuse

langfuse = langfuse.Langfuse()

def generate_trace_id_code():
    # Generate a timestamp-based prefix (YYYYMMDDHHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Calculate remaining characters needed (20 - 14 = 6)
    remaining_length = 20 - len(timestamp)

    # Generate random alphanumeric characters for the remaining length
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=remaining_length))

    # Combine timestamp and random part
    trace_id = timestamp + random_part
    return trace_id

def score_response_async(trace_id, feedback):
    """Run feedback submission in a background thread to avoid blocking UI."""
    def _submit_score():
        langfuse.score(
            trace_id=trace_id,
            name="correctness",
            value=feedback
        )
    
    thread = threading.Thread(target=_submit_score)
    thread.start()
