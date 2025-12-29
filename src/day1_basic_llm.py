from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def ask_llm(prompt: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )
    result = llm.invoke(prompt)
    return result.content.strip()


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Add it to .env and try again.")

    prompt = "Explain LangChain in 3 bullet points."
    answer = ask_llm(prompt)

    print("\nPrompt:")
    print(prompt)
    print("\nAnswer:")
    print(answer)
    print()


if __name__ == "__main__":
    main()
