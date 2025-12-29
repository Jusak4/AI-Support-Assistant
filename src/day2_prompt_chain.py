from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class SupportInput:
    app_name: str
    user_issue: str


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful technical support assistant. "
                "Be concise, practical, and specific. If you make assumptions, label them.",
            ),
            (
                "human",
                "App: {app_name}\n"
                "User issue: {user_issue}\n\n"
                "Return the answer with these headings:\n"
                "1) Problem Understanding\n"
                "2) Possible Cause\n"
                "3) Suggested Steps\n",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Runnable pipeline: prompt -> llm
    return prompt | llm


def parse_args(argv: list[str]) -> SupportInput:
    # Usage:
    # python src/day2_prompt_chain.py "CloudAppX" "I can't log in; it says invalid token"
    if len(argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python src/day2_prompt_chain.py <app_name> <user_issue>\n\n"
            "Example:\n"
            "  python src/day2_prompt_chain.py CloudAppX \"Login fails with invalid token\""
        )

    app_name = argv[0].strip()
    user_issue = " ".join(argv[1:]).strip()
    return SupportInput(app_name=app_name, user_issue=user_issue)


def main(argv: list[str]) -> int:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Add it to your .env file.")
        return 1

    data = parse_args(argv)
    chain = build_chain()

    result = chain.invoke({"app_name": data.app_name, "user_issue": data.user_issue})

    print("\n--- Structured Support Answer ---\n")
    print(result.content.strip())
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
