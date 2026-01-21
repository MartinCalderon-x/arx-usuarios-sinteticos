# Product Seed

**[English](#english) | [Español](#español)**

---

<a name="english"></a>

# English

> Your AI-powered product factory. Build anything with intelligent tools.

## The Story: The Factory and The Product

Imagine you're an architect. You have a fully-equipped workshop with the best tools: laser cutters, 3D printers, AI assistants that understand blueprints. This workshop doesn't ship to your clients - it stays with you. What ships is the beautiful furniture, the custom doors, the innovative products you create with those tools.

**Product Seed works the same way.**

```
product-seed/
├── factory/     ← Your AI workshop (stays with you)
│   └── Arx-SDK: LLM, Agents, RAG, Memory
│
└── product/     ← What you build (gets deployed)
    └── React, Python, Go, Rust... anything
```

The **factory** uses [Arx-SDK](https://github.com/MartinCalderon-x/arx-sdk) to give you:
- Language models that write and analyze code
- AI agents that design, plan, and implement
- Vector search for your documentation
- Global memory across all your projects

The **product** is whatever you're building. A React dashboard. A Python API. A mobile app. The factory helps you build it, but the product is technology-agnostic.

---

## Getting Started

### Step 1: Clone Your Factory (2 minutes)

```bash
git clone git@github.com:MartinCalderon-x/product-seed.git my-project
cd my-project
./factory/scripts/setup.sh
```

Done. Your AI workshop is ready.

### Step 2: Configure Your Tools (5 minutes)

```bash
# Edit .env with your API keys
vim .env
```

```bash
# Minimum: one LLM provider
ANTHROPIC_API_KEY=sk-ant-...

# Optional: global memory
ARX_CODEX_URL=https://your-project.supabase.co
ARX_CODEX_SECRET_KEY=sb_secret_...
```

### Step 3: Test Your Factory

```bash
source venv/bin/activate
python factory/run.py
```

You should see your tools are configured and ready.

### Step 4: Build Your Product

Now create whatever you want in `product/`:

**React + Vite:**
```bash
cd product
npm create vite@latest . -- --template react-ts
```

**Python API:**
```bash
cd product
mkdir src
touch src/__init__.py src/main.py requirements.txt
```

**Next.js:**
```bash
cd product
npx create-next-app@latest .
```

### Step 5: Use Arx-SDK to Build

Create scripts that use the AI tools to help build your product:

```python
# scripts/generate_feature.py
from dotenv import load_dotenv
load_dotenv()

from arx_llm import get_llm
from arx_agents import DeveloperAgent
from arx_codex import ArxCodexClient

# Initialize
llm = get_llm()
codex = ArxCodexClient()

# Search for existing patterns
patterns = codex.search("authentication JWT")
print(f"Found {len(patterns)} relevant patterns")

# Use an agent to implement
developer = DeveloperAgent(llm=llm, codex=codex)
result = developer.implement_feature(
    "JWT authentication with refresh tokens",
    language="python",
    framework="fastapi"
)

print(result["code"])
```

```python
# scripts/index_docs.py
from arx_rag import RAGSystem, load_documents

rag = RAGSystem(persist_dir="./data/vectordb")
docs = load_documents("./product/docs", chunk_size=500)
rag.add_documents(docs)

# Now you can query
result = rag.query("How does the auth system work?")
print(result.answer)
```

### Step 6: Deploy Your Product

The Dockerfile only deploys `product/` - customize it for your technology:

```bash
# Build
docker build -t my-product .

# Run locally
docker run -p 8080:8080 my-product

# Deploy (automatic via GitHub Actions)
git push origin develop  # → DEV
git tag v1.0.0 && git push --tags  # → PROD
```

---

## Project Structure

```
my-project/
├── factory/                 # AI Workshop (not deployed)
│   ├── scripts/
│   │   └── setup.sh         # One-command setup
│   ├── run.py               # Test configuration
│   └── requirements.txt     # Arx-SDK
│
├── product/                 # Your Product (deployed)
│   └── .gitkeep             # Add your code here
│
├── data/                    # Factory runtime data
│   ├── vectordb/            # ChromaDB (RAG)
│   ├── cache/               # LLM cache
│   └── logs/
│
├── data-test/               # Test fixtures
│   └── samples/
│
├── tests/                   # Your product tests
│
├── scripts/                 # Your build scripts (optional)
│
├── .github/workflows/
│   └── deploy.yml           # Cloud Run CI/CD
│
├── Dockerfile               # Builds product/ only
├── .env.example             # Configuration template
└── README.md
```

## Arx-SDK Packages

| Package | Import | Description |
|---------|--------|-------------|
| arx-llm | `from arx_llm import get_llm` | Any LLM with one interface |
| arx-agents | `from arx_agents import DeveloperAgent` | AI agents that code |
| arx-codex | `from arx_codex import ArxCodexClient` | Global memory |
| arx-rag | `from arx_rag import RAGSystem` | Document search |
| arx-intelligence | `from arx_intelligence import TechStackIntelligence` | Analysis tools |

## Commands

```bash
# Setup factory
./factory/scripts/setup.sh

# Activate environment
source venv/bin/activate

# Test factory tools
python factory/run.py

# Run tests
pytest tests/

# Build Docker image
docker build -t my-product .
```

---

<a name="español"></a>

# Español

> Tu fábrica de productos con IA. Construye cualquier cosa con herramientas inteligentes.

## La Historia: La Fábrica y El Producto

Imagina que eres un arquitecto. Tienes un taller completamente equipado con las mejores herramientas: cortadoras láser, impresoras 3D, asistentes de IA que entienden planos. Este taller no se envía a tus clientes - se queda contigo. Lo que se envía son los muebles hermosos, las puertas personalizadas, los productos innovadores que creas con esas herramientas.

**Product Seed funciona igual.**

```
product-seed/
├── factory/     ← Tu taller de IA (se queda contigo)
│   └── Arx-SDK: LLM, Agentes, RAG, Memoria
│
└── product/     ← Lo que construyes (se despliega)
    └── React, Python, Go, Rust... cualquier cosa
```

La **fábrica** usa [Arx-SDK](https://github.com/MartinCalderon-x/arx-sdk) para darte:
- Modelos de lenguaje que escriben y analizan código
- Agentes de IA que diseñan, planifican e implementan
- Búsqueda vectorial para tu documentación
- Memoria global entre todos tus proyectos

El **producto** es lo que estés construyendo. Un dashboard React. Una API Python. Una app móvil. La fábrica te ayuda a construirlo, pero el producto es agnóstico a la tecnología.

---

## Comenzando

### Paso 1: Clona Tu Fábrica (2 minutos)

```bash
git clone git@github.com:MartinCalderon-x/product-seed.git mi-proyecto
cd mi-proyecto
./factory/scripts/setup.sh
```

Listo. Tu taller de IA está preparado.

### Paso 2: Configura Tus Herramientas (5 minutos)

```bash
# Edita .env con tus API keys
vim .env
```

```bash
# Mínimo: un proveedor de LLM
ANTHROPIC_API_KEY=sk-ant-...

# Opcional: memoria global
ARX_CODEX_URL=https://tu-proyecto.supabase.co
ARX_CODEX_SECRET_KEY=sb_secret_...
```

### Paso 3: Prueba Tu Fábrica

```bash
source venv/bin/activate
python factory/run.py
```

Deberías ver que tus herramientas están configuradas y listas.

### Paso 4: Construye Tu Producto

Ahora crea lo que quieras en `product/`:

**React + Vite:**
```bash
cd product
npm create vite@latest . -- --template react-ts
```

**Python API:**
```bash
cd product
mkdir src
touch src/__init__.py src/main.py requirements.txt
```

**Next.js:**
```bash
cd product
npx create-next-app@latest .
```

### Paso 5: Usa Arx-SDK para Construir

Crea scripts que usen las herramientas de IA para construir tu producto:

```python
# scripts/generate_feature.py
from dotenv import load_dotenv
load_dotenv()

from arx_llm import get_llm
from arx_agents import DeveloperAgent
from arx_codex import ArxCodexClient

# Inicializar
llm = get_llm()
codex = ArxCodexClient()

# Buscar patrones existentes
patterns = codex.search("authentication JWT")
print(f"Encontrados {len(patterns)} patrones relevantes")

# Usar un agente para implementar
developer = DeveloperAgent(llm=llm, codex=codex)
result = developer.implement_feature(
    "Autenticación JWT con refresh tokens",
    language="python",
    framework="fastapi"
)

print(result["code"])
```

```python
# scripts/index_docs.py
from arx_rag import RAGSystem, load_documents

rag = RAGSystem(persist_dir="./data/vectordb")
docs = load_documents("./product/docs", chunk_size=500)
rag.add_documents(docs)

# Ahora puedes consultar
result = rag.query("¿Cómo funciona el sistema de auth?")
print(result.answer)
```

### Paso 6: Despliega Tu Producto

El Dockerfile solo despliega `product/` - customízalo para tu tecnología:

```bash
# Build
docker build -t mi-producto .

# Correr localmente
docker run -p 8080:8080 mi-producto

# Deploy (automático via GitHub Actions)
git push origin develop  # → DEV
git tag v1.0.0 && git push --tags  # → PROD
```

---

## Estructura del Proyecto

```
mi-proyecto/
├── factory/                 # Taller de IA (no se despliega)
│   ├── scripts/
│   │   └── setup.sh         # Setup con un comando
│   ├── run.py               # Probar configuración
│   └── requirements.txt     # Arx-SDK
│
├── product/                 # Tu Producto (se despliega)
│   └── .gitkeep             # Agrega tu código aquí
│
├── data/                    # Datos runtime de la fábrica
│   ├── vectordb/            # ChromaDB (RAG)
│   ├── cache/               # Cache de LLM
│   └── logs/
│
├── data-test/               # Fixtures de test
│   └── samples/
│
├── tests/                   # Tests de tu producto
│
├── scripts/                 # Tus scripts de build (opcional)
│
├── .github/workflows/
│   └── deploy.yml           # CI/CD para Cloud Run
│
├── Dockerfile               # Construye solo product/
├── .env.example             # Template de configuración
└── README.md
```

## Paquetes Arx-SDK

| Paquete | Import | Descripción |
|---------|--------|-------------|
| arx-llm | `from arx_llm import get_llm` | Cualquier LLM con una interfaz |
| arx-agents | `from arx_agents import DeveloperAgent` | Agentes IA que programan |
| arx-codex | `from arx_codex import ArxCodexClient` | Memoria global |
| arx-rag | `from arx_rag import RAGSystem` | Búsqueda de documentos |
| arx-intelligence | `from arx_intelligence import TechStackIntelligence` | Herramientas de análisis |

## Comandos

```bash
# Setup de la fábrica
./factory/scripts/setup.sh

# Activar entorno
source venv/bin/activate

# Probar herramientas
python factory/run.py

# Correr tests
pytest tests/

# Build imagen Docker
docker build -t mi-producto .
```

---

## License / Licencia

MIT - Use freely. / Úsalo libremente.
