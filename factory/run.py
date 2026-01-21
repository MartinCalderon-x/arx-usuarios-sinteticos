"""
Factory Runner

Verify your AI tools are configured correctly.
Run with: python factory/run.py
"""
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()


def main():
    """Factory configuration check."""
    print("=" * 50)
    print("  Usuarios Sintéticos - Factory Tools")
    print("=" * 50)
    print()

    # Check configuration
    print("Configuration status:")
    print()

    # Factory LLM (Arx-SDK)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("  [OK] Anthropic API key configured (Factory)")
    elif os.getenv("OPENAI_API_KEY"):
        print("  [OK] OpenAI API key configured (Factory)")
    elif os.getenv("LLM_PROVIDER") == "ollama":
        print("  [OK] Ollama configured (Factory, local)")
    else:
        print("  [--] Factory LLM not configured (optional)")

    # Arx-Codex
    if os.getenv("ARX_CODEX_URL") and os.getenv("ARX_CODEX_SECRET_KEY"):
        print("  [OK] Arx-Codex configured (global memory)")
    else:
        print("  [--] Arx-Codex not configured (optional)")

    # ChromaDB
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/vectordb")
    print(f"  [OK] ChromaDB will persist to: {chroma_dir}")

    print()
    print("-" * 50)
    print("  Product Configuration (Usuarios Sintéticos)")
    print("-" * 50)
    print()

    # Gemini (Primary AI for product)
    if os.getenv("GOOGLE_API_KEY"):
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        print(f"  [OK] Gemini API configured (model: {model})")
    else:
        print("  [!!] Gemini not configured - set GOOGLE_API_KEY")

    # Qwen (Alternative Vision)
    if os.getenv("QWEN_API_KEY"):
        print("  [OK] Qwen 2.5 VL configured (alternative vision)")
    else:
        print("  [--] Qwen 2.5 VL not configured (placeholder)")

    # Supabase
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        print("  [OK] Supabase configured (database + auth)")
    else:
        print("  [!!] Supabase not configured - set SUPABASE_URL and keys")

    print()
    print("=" * 50)
    print("  Available Arx-SDK Packages")
    print("=" * 50)
    print()
    print("  arx_llm        - Multi-LLM provider")
    print("  arx_codex      - Global memory (Supabase)")
    print("  arx_agents     - AI Agents (Designer, TechLead, Developer)")
    print("  arx_rag        - RAG with ChromaDB")
    print("  arx_intelligence - Tech stack & design analysis")
    print()
    print("=" * 50)
    print("  Usage Examples")
    print("=" * 50)
    print()
    print("  # LLM")
    print("  from arx_llm import get_llm")
    print("  llm = get_llm()")
    print("  response = llm.invoke('Hello!')")
    print()
    print("  # Agents")
    print("  from arx_agents import DeveloperAgent")
    print("  dev = DeveloperAgent(llm=llm)")
    print("  code = dev.implement_feature('JWT auth', language='python')")
    print()
    print("  # RAG")
    print("  from arx_rag import RAGSystem, load_documents")
    print("  rag = RAGSystem()")
    print("  rag.add_documents(load_documents('./docs'))")
    print("  result = rag.query('How does auth work?')")
    print()
    print("  # Global Memory")
    print("  from arx_codex import ArxCodexClient")
    print("  codex = ArxCodexClient()")
    print("  patterns = codex.search('authentication patterns')")
    print()
    print("Ready to build! Use these imports in your scripts.")
    print()


if __name__ == "__main__":
    main()
