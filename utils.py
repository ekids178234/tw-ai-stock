"""Shared helpers."""
import os
from pathlib import Path

def get_token():
    # env
    tok = os.getenv("FINMIND_TOKEN")
    if tok:
        return tok
    # streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "FINMIND_TOKEN" in st.secrets:
            return st.secrets["FINMIND_TOKEN"]
    except Exception:
        pass
    # .env file
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
        tok = os.getenv("FINMIND_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return None