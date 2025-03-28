from utils.env_setup import client
from utils.helpers import generate_trace_id_code

def generate_response(context, query):
    """Generates a response using OpenAI models."""
    trace_id = generate_trace_id_code()  # Generate a new trace ID

    response = client.chat.completions.create(
        model="o3-mini",  # Ensure the correct OpenAI model is used
        messages=[
            {"role": "system", "content": "You should always reply in spanish. You are an AI assistant that provides helpful responses based on retrieved documents. Ignore any previous knowledge, just stick to the context given. If the answer needs to include a table, please format your answer as a Markdown table, using a header row, separator row, and separate rows for each field. Do not specify details about the chunks used to answer."},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ],

        trace_id=trace_id  # Pass the generated trace ID
    )
    
    return response.choices[0].message.content.strip(), trace_id  # Return trace_id as 
