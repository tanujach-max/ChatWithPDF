from groq import Groq
from typing import List, Dict

def get_groq_answer(
    question: str,
    relevant_chunks: List[str],
    api_key: str,
    chat_history: List[Dict] = []
) -> str:
    client = Groq(api_key=api_key)

    context = "\n\n---\n\n".join(relevant_chunks)

    history_text = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    system_prompt = f"""You are a helpful assistant that answers questions based on the provided PDF document content.

INSTRUCTIONS:
- Answer ONLY based on the PDF content provided below
- If the answer is not in the PDF, say "I couldn't find this information in the PDF"
- Be clear, concise, and helpful
- If relevant, mention which part of the document the answer comes from

PDF CONTENT (most relevant sections):
{context}

PREVIOUS CONVERSATION:
{history_text}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        error = str(e)
        if "invalid_api_key" in error.lower() or "401" in error:
            return "❌ Invalid API key. Please check your Groq API key."
        elif "rate_limit" in error.lower() or "429" in error:
            return "❌ Rate limit reached. Please wait a moment and try again."
        else:
            raise Exception(f"Groq API error: {error}")