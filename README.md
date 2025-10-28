# FloatChat 🌊

**FloatChat** is an AI-powered conversational data explorer for ARGO ocean data. This Streamlit application leverages a Retrieval-Augmented Generation (RAG) pipeline with Google's Gemini API to allow users to find, visualize, and understand complex oceanographic data using natural language.

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 🎥 Demo

*(It is highly recommended to add a quick GIF or screenshot of the app in action here.)*

---

## 🎯 Key Features

* **Conversational Querying:** Ask questions in plain English (e.g., "Show me salinity profiles near the equator in March 2023").
* **AI-Powered RAG:** Uses a **ChromaDB** vector store and **SentenceTransformers** to perform semantic search, finding the most relevant ARGO floats for your query from a persistent database.
* **Gemini-Driven Analysis:** Integrates Google's **Gemini API** for multiple tasks:
    * **Query Planning:** Converts your natural language query into a structured JSON plan for execution.
    * **Automated Visualization:** Automatically generates the best plot for your query (geographic maps, vertical profiles, or time series).
    * **Natural Language Insights:** Provides a detailed, AI-generated oceanographic analysis of the visualized data.
    * **Anomaly Detection:** Identifies statistical anomalies in time series data and uses AI to suggest plausible oceanographic causes (e.g., eddies, blooms, instrument error).
* **Comparative Analysis:** A "Compare Slots" feature lets you save two different datasets and generate a side-by-side visual and AI-powered comparison.
* **Persistent & Scalable:** Uses **PostgreSQL** with the **PostGIS** extension for robust, spatially-aware storage of ARGO data.
* **Multiple Data Sources:**
    * **Smart Search (RAG):** Query the persistent database using natural language.
    * **PostgreSQL Database:** Manually select floats from the existing database.
    * **Live Argopy Fetch:** Pull live physical (`phy`) or biogeochemical (`bgc`) data directly from ARGO data centers.
* **Chat Export:** Export your conversation history, including AI analysis, as a PDF document.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI & LLM:** [Google Gemini API](https://ai.google.dev/)
* **RAG Pipeline:**
    * **Vector Database:** [ChromaDB](https://www.trychroma.com/)
    * **Embedding Model:** [SentenceTransformers](https://sbert.net/) (`all-MiniLM-L6-v2`)
* **Ocean Data:** [Argopy](https://argopy.readthedocs.io/en/latest/)
* **Database:** [PostgreSQL](https://www.postgresql.org/) with [PostGIS](https://postgis.net/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/), [Xarray](https://xarray.dev/)
* **Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/)
* **NLP (Entity Extraction):** [spaCy](https://spacy.io/)

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10+
* Access to a running PostgreSQL database (v14+) with the PostGIS extension enabled.
* A Google Gemini API Key.
* A `requirements.txt` file in your project root (see setup instructions).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/FloatChat.git](https://github.com/your-username/FloatChat.git)
    cd FloatChat
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Download the spaCy NLP model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

### 3. Configuration (Secrets)

Create a `.streamlit/secrets.toml` file to store your API keys and database credentials. **Do not commit this file to Git.**

```toml
# .streamlit/secrets.toml

# Google Gemini API Key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# PostgreSQL Database Configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "argo_oceandb"
DB_USER = "floatchat_user"
DB_PASSWORD = "YOUR_DB_PASSWORD"
