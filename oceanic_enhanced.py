# oceanic_enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xarray as xr
import warnings
import spacy
from spacy.matcher import PhraseMatcher
import requests
import json
import re
from scipy.stats import linregress
import cftime
from pandas.api import types as ptypes
from datetime import datetime
import logging

# New: argopy for Argo data access
import argopy
from argopy import DataFetcher as ArgoDataFetcher

# New: embed Euro-Argo Data Selection Tool
import streamlit.components.v1 as components

# --- Database imports ---
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import psycopg2.pool
from psycopg2 import sql

# --- NEW: RAG COMPONENTS ---
import chromadb
from sentence_transformers import SentenceTransformer

# --- NEW: GEMINI API INTEGRATION ---
import google.generativeai as genai

# Add these imports after your existing imports
from datetime import timedelta
import dateparser
from geopy.distance import geodesic
from plotly.subplots import make_subplots
from psycopg2.extensions import register_adapter, AsIs
from fpdf import FPDF
from utils.globe import plot_floats_on_globe

# Adapters for NumPy data types
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

# Register the adapters with psycopg2
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.int32, addapt_numpy_int32)


class PDF(FPDF):
    """Custom PDF class to handle headers and footers."""
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'FloatChat Conversation Export', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_from_history(selected_chats: list) -> bytes:
    """Generates a PDF file from a list of selected chat pairs and returns it as bytes."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Helvetica', '', 11)

    if not selected_chats:
        pdf.multi_cell(0, 10, 'No conversations were selected for export.')
        return pdf.output(dest='S').encode('latin-1')

    for chat_pair in selected_chats:
        user_msg = chat_pair['user']
        assistant_msg = chat_pair['assistant']

        # -- User Query --
        pdf.set_font('Helvetica', 'B', 12)
        pdf.multi_cell(0, 10, '>> Your Query:')
        pdf.set_font('Helvetica', '', 11)
        # Use multi_cell to handle automatic line breaks for long text
        pdf.multi_cell(0, 8, user_msg['content'])
        pdf.ln(5)

        # -- Assistant Response --
        pdf.set_font('Helvetica', 'B', 12)
        pdf.multi_cell(0, 10, '<< AI Assistant\'s Response:')
        pdf.set_font('Helvetica', '', 11)
        pdf.multi_cell(0, 8, assistant_msg['content'])

        # -- Separator for next entry --
        pdf.ln(10)
        # Draw a horizontal line as a separator
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(10)

    # Output the PDF as a byte string
    return pdf.output(dest='S').encode('latin-1')

def create_pdf_from_comparison(title: str, content: str) -> bytes:
    """Generates a simple PDF from a title and a block of text content."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.multi_cell(0, 10, title, 0, 1, 'C')
    pdf.ln(10)

    # Add content - Note: This will not render markdown tables perfectly but will preserve the text
    pdf.set_font('Helvetica', '', 11)
    # Replace markdown table separators with simple dashes for better text flow in PDF
    cleaned_content = re.sub(r'\| :--- \|', '------------------', content)
    cleaned_content = cleaned_content.replace('|', ' ')
    pdf.multi_cell(0, 8, cleaned_content)

    return pdf.output(dest='S').encode('latin-1')

# === PostgreSQL helper layer ===
@st.cache_resource
def load_database_config():
    """Return database config; falls back to sensible defaults."""
    return {
        "host":      st.secrets.get("DB_HOST", "localhost"),
        "database":  st.secrets.get("DB_NAME", "argo_oceandb"),
        "user":      st.secrets.get("DB_USER", "floatchat_user"),
        "password":  st.secrets.get("DB_PASSWORD", "140612"),
        "port":      int(st.secrets.get("DB_PORT", 5432)),
        "minconn":   1,
        "maxconn":   20,
        "connect_timeout": 1000,
        "application_name": "FloatChat",
        "sslmode":   "prefer",
        "client_encoding": "UTF8",
    }

@st.cache_resource
def init_database_pool():
    """Initialise a threaded connection-pool once per session."""
    cfg = load_database_config()
    try:
        pool = psycopg2.pool.ThreadedConnectionPool(
            cfg["minconn"], cfg["maxconn"],
            host=cfg["host"], database=cfg["database"],
            user=cfg["user"], password=cfg["password"],
            port=cfg["port"], connect_timeout=cfg["connect_timeout"],
            application_name=cfg["application_name"],
            sslmode=cfg["sslmode"], client_encoding=cfg["client_encoding"],
        )
        return pool
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def _get_conn(pool):
    return pool.getconn() if pool else None

def _put_conn(pool, conn):
    if pool and conn:
        pool.putconn(conn)

def create_database_schema(_pool):
    """Create or update the database schema with full spatial enhancements."""
    if not _pool:
        return False

    # Basic schema first
    basic_schema = [
        """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id SERIAL PRIMARY KEY,
            platform_number VARCHAR(50) NOT NULL,
            cycle_number INTEGER NOT NULL,
            latitude FLOAT,
            longitude FLOAT,
            datetime TIMESTAMP,
            juld FLOAT,
            position_qc INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(platform_number, cycle_number)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS measurements (
            measurement_id SERIAL PRIMARY KEY,
            profile_id INTEGER REFERENCES profiles(profile_id) ON DELETE CASCADE,
            parameter_name VARCHAR(50) NOT NULL,
            pressure FLOAT,
            value FLOAT,
            value_adjusted FLOAT,
            qc_flag INTEGER DEFAULT 1,
            data_mode VARCHAR(1) DEFAULT 'R',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS bgc_parameters (
            parameter_id SERIAL PRIMARY KEY,
            parameter_name VARCHAR(50) UNIQUE NOT NULL,
            long_name TEXT,
            units VARCHAR(50),
            description TEXT,
            category VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]

    # Enhanced spatial schema
    spatial_schema = []
    conn = None

    try:
        conn = _get_conn(_pool)

        # Execute basic schema first
        with conn.cursor() as cursor:
            for statement in basic_schema:
                cursor.execute(statement)

        # Try PostGIS enhancements
        try:
            with conn.cursor() as cursor:
                # Enable PostGIS
                cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

                # Add spatial column
                cursor.execute("""
                    ALTER TABLE profiles
                    ADD COLUMN IF NOT EXISTS location_point GEOMETRY(POINT, 4326);
                """)

                # Create spatial indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_profiles_location
                    ON profiles USING GIST(location_point);
                """)

                # Create trigger function
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_profile_location()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.location_point = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)

                # Create trigger
                cursor.execute("""
                    DROP TRIGGER IF EXISTS trigger_update_location ON profiles;
                    CREATE TRIGGER trigger_update_location
                    BEFORE INSERT OR UPDATE ON profiles
                    FOR EACH ROW EXECUTE FUNCTION update_profile_location();
                """)

                # Create ocean regions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ocean_regions (
                        region_id SERIAL PRIMARY KEY,
                        region_name VARCHAR(100) UNIQUE NOT NULL,
                        boundary GEOMETRY(POLYGON, 4326),
                        description TEXT
                    );
                """)

                # Insert predefined regions
                cursor.execute("""
                INSERT INTO ocean_regions (region_name, boundary, description) VALUES
                ('Arabian Sea', ST_GeomFromText('POLYGON((50 8, 78 8, 78 28, 50 28, 50 8))', 4326), 'Arabian Sea region'),
                ('Equatorial Band', ST_GeomFromText('POLYGON((-180 -5, 180 -5, 180 5, -180 5, -180 -5))', 4326), 'Equatorial region ±5°'),
                ('Indian Ocean', ST_GeomFromText('POLYGON((20 -40, 120 -40, 120 30, 20 30, 20 -40))', 4326), 'Indian Ocean basin'),
                ('Pacific Ocean', ST_GeomFromText('POLYGON((120 -60, -70 -60, -70 65, 120 65, 120 -60))', 4326), 'Pacific Ocean basin'),
                ('Atlantic Ocean', ST_GeomFromText('POLYGON((-80 -60, 20 -60, 20 70, -80 70, -80 -60))', 4326), 'Atlantic Ocean basin'),
                ('Mediterranean Sea', ST_GeomFromText('POLYGON((-6 30, 36 30, 36 46, -6 46, -6 30))', 4326), 'Mediterranean Sea'),
                ('Red Sea', ST_GeomFromText('POLYGON((32 12, 43 12, 43 30, 32 30, 32 12))', 4326), 'Red Sea'),
                ('South China Sea', ST_GeomFromText('POLYGON((99 0, 125 0, 125 25, 99 25, 99 0))', 4326), 'South China Sea'),
                ('Caribbean Sea', ST_GeomFromText('POLYGON((-85 9, -60 9, -60 25, -85 25, -85 9))', 4326), 'Caribbean Sea'),
                ('Gulf of Mexico', ST_GeomFromText('POLYGON((-98 18, -80 18, -80 31, -98 31, -98 18))', 4326), 'Gulf of Mexico')
                ON CONFLICT (region_name) DO NOTHING;
                """)


                # ADD this new table creation after the existing ocean_regions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS enhanced_ocean_regions (
                        region_id SERIAL PRIMARY KEY,
                        region_name VARCHAR(150) UNIQUE NOT NULL,
                        region_type VARCHAR(50) NOT NULL, -- 'basin', 'sea', 'ecoregion', 'lme', 'water_mass', 'biogeochemical'
                        boundary GEOMETRY(POLYGON, 4326),
                        depth_min FLOAT,
                        depth_max FLOAT,
                        characteristics JSONB,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # INSERT expanded region definitions
                enhanced_regions = [
                    # Major Ocean Basins
                    ("Pacific Ocean", "basin", "POLYGON((120 -60, -70 -60, -70 65, 120 65, 120 -60))", None, None,
                     '{"type": "major_basin", "area_km2": 165200000}', "Largest ocean basin covering about 46% of water surface"),
                    ("Atlantic Ocean", "basin", "POLYGON((-80 -60, 20 -60, 20 70, -80 70, -80 -60))", None, None,
                     '{"type": "major_basin", "area_km2": 106460000}', "Second largest ocean basin"),
                    # Regional Seas
                    ("Mediterranean Sea", "sea", "POLYGON((-6 30, 36 30, 36 46, -6 46, -6 30))", 0, 5267,
                     '{"type": "enclosed_sea", "salinity_range": "36-39"}', "Semi-enclosed sea between Europe, Africa and Asia"),
                    ("South China Sea", "sea", "POLYGON((99 0, 125 0, 125 25, 99 25, 99 0))", 0, 5016,
                     '{"type": "marginal_sea", "monsoon_influenced": true}', "Marginal sea in Southeast Asia"),
                    # Marine Ecoregions
                    ("Coral Triangle", "ecoregion", "POLYGON((93 -11, 150 -11, 150 20, 93 20, 93 -11))", 0, 200,
                     '{"biodiversity": "highest", "coral_species": 600}', "Global center of marine biodiversity"),
                    # Biogeochemical Provinces
                    ("Subtropical Gyre - North Pacific", "biogeochemical", "POLYGON((130 15, -130 15, -130 35, 130 35, 130 15))", 0, 2000,
                     '{"productivity": "oligotrophic", "chlorophyll": "low"}', "Low productivity subtropical waters"),
                    # Large Marine Ecosystems
                    ("Benguela Current LME", "lme", "POLYGON((8 -35, 18 -35, 18 -15, 8 -15, 8 -35))", 0, 3000,
                     '{"upwelling": true, "productivity": "high"}', "Major upwelling ecosystem off southwest Africa"),
                    # Water Mass Regions
                    ("North Atlantic Deep Water", "water_mass", "POLYGON((-70 40, -10 40, -10 70, -70 70, -70 40))", 1500, 4000,
                     '{"temperature_range": "1.5-4.0", "salinity_range": "34.8-35.0"}', "Major deep water mass"),
                ]

                for name, rtype, boundary, dmin, dmax, chars, desc in enhanced_regions:
                    cursor.execute("""
                        INSERT INTO enhanced_ocean_regions (region_name, region_type, boundary, depth_min, depth_max, characteristics, description)
                        VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s)
                        ON CONFLICT (region_name) DO NOTHING;
                    """, (name, rtype, boundary, dmin, dmax, chars, desc))

                # Create profile parameters table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS profile_parameters (
                        profile_id INTEGER REFERENCES profiles(profile_id),
                        parameter_name VARCHAR(50) NOT NULL,
                        min_pressure FLOAT,
                        max_pressure FLOAT,
                        measurement_count INTEGER,
                        data_quality_score FLOAT DEFAULT 1.0,
                        PRIMARY KEY (profile_id, parameter_name)
                    );
                """)

                # Additional indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_profiles_datetime ON profiles(datetime);
                    CREATE INDEX IF NOT EXISTS idx_profiles_region ON profiles(latitude, longitude);
                    CREATE INDEX IF NOT EXISTS idx_measurements_param_pressure ON measurements(parameter_name, pressure);
                    CREATE INDEX IF NOT EXISTS idx_measurements_profile_param ON measurements(profile_id, parameter_name);
                """)

                st.success("✅ Full enhanced schema with PostGIS created successfully")

        except Exception as spatial_error:
            st.warning(f"⚠️ Spatial enhancements failed: {spatial_error}")
            st.info("Using basic schema without spatial features")

        conn.commit()
        logging.info("Database schema created successfully.")
        return True

    except Exception as e:
        logging.error(f"Failed to create database schema: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            _put_conn(_pool, conn)


@st.cache_resource
def init_chromadb_client():
    """Initialize and return a persistent ChromaDB client."""
    try:
        # Assumes your vector DB is in a local folder named 'chroma_argo_db'
        client = chromadb.PersistentClient(path="./chroma_argo_db")
        return client
    except Exception as e:
        st.error(f"ChromaDB client initialization failed: {e}")
        return None

@st.cache_resource
def get_vector_collection(_client):
    """Get the ARGO metadata collection from ChromaDB."""
    if not _client:
        return None
    try:
        collection = _client.get_or_create_collection(
            name="argo_profiles_metadata",
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        st.error(f"Failed to get ChromaDB collection: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


def store_data_in_database(df: pd.DataFrame, pool):
    """Bulk-insert only the rows currently displayed in df which already honours any float-ID filtering."""
    if df.empty or not pool:
        return False

    conn = _get_conn(pool)
    if not conn:
        return False

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ---- Insert/Upsert profiles ----
            profiles = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER']).first().reset_index()
            profile_rows = [
                (str(r['PLATFORM_NUMBER']), r['CYCLE_NUMBER'],
                 r.get('LATITUDE'), r.get('LONGITUDE'),
                 r.get('DATETIME'), r.get('JULD'), 1)  # default QC=1
                for _, r in profiles.iterrows()
            ]

            if profile_rows:
                execute_values(cur, """
                    INSERT INTO profiles (platform_number, cycle_number, latitude, longitude, datetime, juld, position_qc)
                    VALUES %s
                    ON CONFLICT (platform_number, cycle_number)
                    DO UPDATE SET latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude, datetime=EXCLUDED.datetime, juld=EXCLUDED.juld
                """, profile_rows)

                # Get profile IDs for the inserted/updated profiles using an IN clause
                platform_cycle_pairs = tuple([(p[0], p[1]) for p in profile_rows])
                if not platform_cycle_pairs:
                    profile_id_map = {}
                else:
                    cur.execute("""
                        SELECT profile_id, platform_number, cycle_number
                        FROM profiles
                        WHERE (platform_number, cycle_number) IN %s
                    """, (platform_cycle_pairs,))

                    profile_id_map = {(row['platform_number'], row['cycle_number']): row['profile_id']
                                    for row in cur.fetchall()}

                # ---- Insert measurements ----
                param_cols = ['TEMP', 'PSAL', 'PRES', 'DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL',
                              'BBP700', 'DOWNWELLING_PAR', 'DOWN_IRRADIANCE380',
                              'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE490']

                meas_rows = []
                for _, r in df.iterrows():
                    key = (str(r['PLATFORM_NUMBER']), r['CYCLE_NUMBER'])
                    pid = profile_id_map.get(key)
                    if not pid: continue
                    pressure_val = r.get('PRES')
                    if pd.isna(pressure_val): continue

                    for param in param_cols:
                        if param in df.columns and pd.notna(r.get(param)):
                            adjusted_col = f"{param}_ADJUSTED"
                            qc_col = f"{param}_QC"
                            mode_col = f"{param}_DATA_MODE"
                            meas_rows.append((
                                pid, param, pressure_val, r.get(param),
                                r.get(adjusted_col, r.get(param)),
                                r.get(qc_col, 1),
                                r.get(mode_col, 'R')
                            ))

                if meas_rows:
                    execute_values(cur, """
                        INSERT INTO measurements (profile_id, parameter_name, pressure, value, value_adjusted, qc_flag, data_mode)
                        VALUES %s ON CONFLICT DO NOTHING
                    """, meas_rows)

            conn.commit()
            return True

    except Exception as e:
        logging.error(f"DB insert failed: {e}")
        if conn: conn.rollback()
        return False
    finally:
        _put_conn(pool, conn)

def find_relevant_floats_via_enhanced_rag(query: str, _model, _collection, n_results: int = 30) -> tuple:
    """Enhanced RAG with geographic and temporal awareness."""
    if not query or _model is None or _collection is None:
        return [], [], {}

    try:
        st.info(f"Performing enhanced semantic search for: \"{query}\"")

        # Extract entities for context (DEFENSIVE DEFAULTS)
        intent_info = detect_enhanced_intent(query) or {}
        entities = intent_info.get('entities', {}) or {}
        geo_entities = entities.get('geographic', {}) or {}
        temporal_entities = entities.get('temporal', {}) or {}

        # Create enhanced query embedding
        enhanced_query = query
        if isinstance(geo_entities, dict) and geo_entities.get('regions'):
            region_terms = [r.get('name', r.get('description', '')) for r in (geo_entities.get('regions') or [])]
            enhanced_query += f" {' '.join([t for t in region_terms if t])}"
        if isinstance(temporal_entities, dict) and (temporal_entities.get('dates') or temporal_entities.get('periods')):
            temporal_terms = []
            for period in temporal_entities.get('periods', []) or []:
                temporal_terms.append(period.get('description', ''))
            for date in temporal_entities.get('dates', []) or []:
                temporal_terms.append(date.get('description', ''))
            temporal_terms = [t for t in temporal_terms if t]
            if temporal_terms:
                enhanced_query += f" {' '.join(temporal_terms)}"

        query_embedding = _model.encode([enhanced_query]).tolist()

        # Optional geographic filter for vector search
        where_clause = {}
        if isinstance(geo_entities, dict) and geo_entities.get('regions'):
            region = (geo_entities.get('regions') or [None])[0]
            if isinstance(region, dict) and region.get('type') == 'equatorial':
                where_clause = {"$and": [{"latitude": {"$gte": -5}}, {"latitude": {"$lte": 5}}]}
            elif isinstance(region, dict) and region.get('type') == 'region' and 'bounds' in region:
                b = region['bounds']
                where_clause = {"$and": [
                    {"latitude": {"$gte": b['lat'][0]}},
                    {"latitude": {"$lte": b['lat'][1]}},
                    {"longitude": {"$gte": b['lon'][0]}},
                    {"longitude": {"$lte": b['lon'][1]}},
                ]}

        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
            where=where_clause if where_clause else None
        )

        # DEFENSIVE RESULTS CHECK
        if (
            not results
            or not isinstance(results, dict)
            or not results.get('ids')
            or not results['ids']
            or not results['ids'][0]
        ):
            st.warning("Enhanced semantic search did not return matching profiles.")
            return [], [], intent_info

        # Extract float IDs and context safely
        found_floats = set()
        for metadata in (results.get('metadatas', [[]])[0] or []):
            if isinstance(metadata, dict) and 'platform_number' in metadata:
                found_floats.add(str(metadata['platform_number']))

        context_docs = results.get('documents', [[]])[0] or []
        st.success(f"✅ Enhanced RAG found {len(found_floats)} relevant float(s) with {intent_info.get('primary_action', 'general_query')} intent")
        return list(found_floats), context_docs, intent_info

    except Exception as e:
        st.error(f"Error during enhanced RAG search: {e}")
        return [], [], {}


@st.cache_data(show_spinner="Querying PostgreSQL database...")
def fetch_data_from_database(_pool, floats_key: tuple, temporal_entities: dict = None) -> pd.DataFrame:
    """Fetch ARGO data from PostgreSQL for selected floats, optionally constrained by time windows."""
    if not _pool or not floats_key:
        return pd.DataFrame()
    conn = _get_conn(_pool)
    if not conn:
        st.error("Could not get a connection from the database pool.")
        return pd.DataFrame()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            base = """
                SELECT
                    p.platform_number, p.cycle_number, p.latitude, p.longitude, p.datetime, p.juld,
                    m.pressure, m.parameter_name, m.value_adjusted
                FROM profiles p
                JOIN measurements m ON p.profile_id = m.profile_id
                WHERE p.platform_number = ANY(%s)
            """
            params = [list(floats_key)]
            # Optional temporal filter: OR across multiple windows
            temporal_clause = ""
            if temporal_entities:
                windows = []
                for w in temporal_entities.get("periods", []) + temporal_entities.get("dates", []):
                    windows.append("p.datetime BETWEEN %s AND %s")
                    params.extend([w["start_date"], w["end_date"]])
                if windows:
                    temporal_clause = " AND (" + " OR ".join(windows) + ")"
            query = base + temporal_clause
            cur.execute(query, params)
            results = cur.fetchall()
            if not results:
                st.warning(f"No data found in the database for floats: {', '.join(map(str, floats_key))}")
                return pd.DataFrame()

            df_long = pd.DataFrame(results)
            st.info("Reconstructing DataFrame from database records...")

            # Separate profile metadata from measurements
            profile_metadata_cols = ['platform_number', 'cycle_number', 'latitude', 'longitude', 'datetime', 'juld']
            profile_metadata = df_long[profile_metadata_cols].drop_duplicates(
                subset=['platform_number', 'cycle_number']
            )

            # Pivot measurements
            df_pivot = df_long.pivot_table(
                index=['platform_number', 'cycle_number', 'pressure'],
                columns='parameter_name',
                values='value_adjusted'
            )
            df_pivot.columns.name = None
            df_pivot = df_pivot.reset_index()

            # Merge pivot with metadata
            df_wide = pd.merge(df_pivot, profile_metadata, on=['platform_number', 'cycle_number'])

            # Normalize columns
            df_wide.rename(columns=lambda c: c.upper(), inplace=True)
            df_wide.rename(columns={'PLATFORM_NUMBER': 'FLOAT_ID'}, inplace=True)
            df_wide['PLATFORM_NUMBER'] = df_wide['FLOAT_ID']

            first_cols = ['FLOAT_ID', 'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'LATITUDE', 'LONGITUDE', 'DATETIME', 'JULD', 'PRES', 'TEMP', 'PSAL']
            other_cols = [c for c in df_wide.columns if c not in first_cols]
            final_cols = [c for c in first_cols if c in df_wide.columns] + sorted(other_cols)
            df_wide = df_wide[final_cols]

            if 'DATETIME' in df_wide.columns:
                df_wide['DATETIME'] = pd.to_datetime(df_wide['DATETIME'], errors='coerce')
                df_wide['DATETIME'] = df_wide['DATETIME'].dt.floor('us')

            return df_wide

    except Exception as e:
        st.error(f"Failed to fetch or process data from database: {e}")
        return pd.DataFrame()
    finally:
        _put_conn(_pool, conn)

@st.cache_data(show_spinner="Fetching available floats from DB...")
def get_available_floats_from_db(_pool):
    """Queries the database for a distinct list of available float IDs."""
    if not _pool:
        return []
    conn = _get_conn(_pool)
    if not conn:
        return []
    try:
        query = "SELECT DISTINCT platform_number FROM profiles ORDER BY platform_number;"
        df = pd.read_sql(query, conn)
        return df['platform_number'].tolist()
    except Exception as e:
        st.error(f"Could not fetch float list from DB: {e}")
        return []
    finally:
        _put_conn(_pool, conn)

@st.cache_data(show_spinner="Fetching defined regions from DB...")
def get_defined_regions_from_db(_pool):
    """Queries the database for a distinct list of all defined region names from both tables."""
    if not _pool:
        return []
    conn = _get_conn(_pool)
    if not conn:
        return []
    try:
        # This updated query uses UNION to combine names from both region tables,
        # ensuring all defined regions are available in the sidebar.
        query = """
            SELECT region_name FROM enhanced_ocean_regions
            UNION
            SELECT region_name FROM ocean_regions
            ORDER BY region_name;
        """
        df = pd.read_sql(query, conn)
        return df['region_name'].tolist()
    except Exception as e:
        # This prevents the app from crashing if the tables don't exist yet
        logging.warning(f"Could not fetch regions from DB, tables might not exist: {e}")
        return []
    finally:
        _put_conn(_pool, conn)

# ===== ADD ALL ENHANCED QUERY FUNCTIONS HERE =====

def extract_geographic_entities(query: str) -> dict:
    """Extract geographic references from natural language queries."""

    # EXPANDED region patterns - ADD THESE NEW DEFINITIONS
    region_patterns = {
        # Existing patterns (keep these)
        r'\b(?:near\s+)?(?:the\s+)?equator(?:ial)?(?:\s+region)?\b': {
            'type': 'equatorial',
            'lat_range': (-5, 5),
            'description': 'Equatorial region ±5°'
        },
        r'\b(?:arabian\s+sea|arab\s+sea)\b': {
            'type': 'region',
            'bounds': {'lat': (8, 28), 'lon': (50, 78)},
            'name': 'Arabian Sea'
        },
        r'\b(?:indian\s+ocean)\b': {
            'type': 'region',
            'bounds': {'lat': (-40, 30), 'lon': (20, 120)},
            'name': 'Indian Ocean'
        },
        r'\b(?:bay\s+of\s+bengal)\b': {
            'type': 'region',
            'bounds': {'lat': (5, 22), 'lon': (80, 95)},
            'name': 'Bay of Bengal'
        },

        # NEW PATTERNS TO ADD:
        # Major Ocean Basins
        r'\b(?:pacific\s+ocean|pacific\s+basin)\b': {
            'type': 'region',
            'bounds': {'lat': (-60, 65), 'lon': (120, -70)},
            'name': 'Pacific Ocean'
        },
        r'\b(?:atlantic\s+ocean|atlantic\s+basin)\b': {
            'type': 'region',
            'bounds': {'lat': (-60, 70), 'lon': (-80, 20)},
            'name': 'Atlantic Ocean'
        },
        r'\b(?:southern\s+ocean|antarctic\s+ocean)\b': {
            'type': 'region',
            'bounds': {'lat': (-80, -40), 'lon': (-180, 180)},
            'name': 'Southern Ocean'
        },
        r'\b(?:arctic\s+ocean)\b': {
            'type': 'region',
            'bounds': {'lat': (65, 90), 'lon': (-180, 180)},
            'name': 'Arctic Ocean'
        },

        # Regional Seas and Water Bodies
        r'\b(?:mediterranean\s+sea)\b': {
            'type': 'region',
            'bounds': {'lat': (30, 46), 'lon': (-6, 36)},
            'name': 'Mediterranean Sea'
        },
        r'\b(?:red\s+sea)\b': {
            'type': 'region',
            'bounds': {'lat': (12, 30), 'lon': (32, 43)},
            'name': 'Red Sea'
        },
        r'\b(?:south\s+china\s+sea)\b': {
            'type': 'region',
            'bounds': {'lat': (0, 25), 'lon': (99, 125)},
            'name': 'South China Sea'
        },
        r'\b(?:caribbean\s+sea)\b': {
            'type': 'region',
            'bounds': {'lat': (9, 25), 'lon': (-85, -60)},
            'name': 'Caribbean Sea'
        },
        r'\b(?:gulf\s+of\s+mexico)\b': {
            'type': 'region',
            'bounds': {'lat': (18, 31), 'lon': (-98, -80)},
            'name': 'Gulf of Mexico'
        },

        # Marine Ecoregions (based on MEOW classification)
        r'\b(?:coral\s+triangle)\b': {
            'type': 'region',
            'bounds': {'lat': (-11, 20), 'lon': (93, 150)},
            'name': 'Coral Triangle'
        },
        r'\b(?:benguela\s+upwelling)\b': {
            'type': 'region',
            'bounds': {'lat': (-35, -15), 'lon': (8, 18)},
            'name': 'Benguela Upwelling'
        },
        r'\b(?:california\s+current)\b': {
            'type': 'region',
            'bounds': {'lat': (23, 48), 'lon': (-130, -115)},
            'name': 'California Current'
        },

        # Biogeochemical Provinces (based on BGC-Argo classification)
        r'\b(?:subtropical\s+gyre)\b': {
            'type': 'region',
            'bounds': {'lat': (15, 35), 'lon': (-180, 180)},
            'name': 'Subtropical Gyre'
        },
        r'\b(?:oligotrophic\s+region)\b': {
            'type': 'biogeochemical',
            'bounds': {'lat': (-40, 40), 'lon': (-180, 180)},
            'name': 'Oligotrophic Waters'
        },
        r'\b(?:upwelling\s+region)\b': {
            'type': 'biogeochemical',
            'lat_range': (-60, 60),
            'description': 'Coastal upwelling regions'
        },

        # Water Mass Regions
        r'\b(?:north\s+atlantic\s+deep\s+water|nadw)\b': {
            'type': 'water_mass',
            'bounds': {'lat': (40, 70), 'lon': (-70, -10)},
            'depth_range': (1500, 4000),
            'name': 'North Atlantic Deep Water'
        },
        r'\b(?:antarctic\s+intermediate\s+water|aaiw)\b': {
            'type': 'water_mass',
            'bounds': {'lat': (-60, -30), 'lon': (-180, 180)},
            'depth_range': (500, 1500),
            'name': 'Antarctic Intermediate Water'
        },

        # Large Marine Ecosystems (LMEs)
        r'\b(?:gulf\s+of\s+alaska\s+lme)\b': {
            'type': 'lme',
            'bounds': {'lat': (52, 62), 'lon': (-170, -130)},
            'name': 'Gulf of Alaska LME'
        },
        r'\b(?:benguela\s+current\s+lme)\b': {
            'type': 'lme',
            'bounds': {'lat': (-35, -15), 'lon': (8, 18)},
            'name': 'Benguela Current LME'
        }
    }


    # Coordinate pattern (e.g., "near 15.5N 65.2E")
    coord_pattern = r'(?:near\s+)?(\d+(?:\.\d+)?)\s*[°]?\s*([NS])\s+(\d+(?:\.\d+)?)\s*[°]?\s*([EW])'

    entities = {'regions': [], 'coordinates': [], 'distances': []}

    query_lower = query.lower()

    # Extract regions
    for pattern, region_info in region_patterns.items():
        if re.search(pattern, query_lower):
            entities['regions'].append(region_info)

    # Extract coordinates
    coord_matches = re.finditer(coord_pattern, query_lower)
    for match in coord_matches:
        lat = float(match.group(1)) * (1 if match.group(2).upper() == 'N' else -1)
        lon = float(match.group(3)) * (1 if match.group(4).upper() == 'E' else -1)
        entities['coordinates'].append({'lat': lat, 'lon': lon})

    # Extract distance terms
    distance_pattern = r'(?:within|near|closest|nearest)(?:\s+(\d+(?:\.\d+)?)\s*(km|kilometers|miles|nm|nautical\s+miles))?'
    distance_matches = re.finditer(distance_pattern, query_lower)
    for match in distance_matches:
        distance_info = {'type': 'proximity'}
        if match.group(1):
            distance_info['value'] = float(match.group(1))
            distance_info['unit'] = match.group(2)
        entities['distances'].append(distance_info)

    return entities

import re
from datetime import datetime, timedelta
import dateparser # Make sure this import is at the top of your file

def extract_temporal_entities(query: str) -> dict:
    """Extract temporal references from natural language queries."""

    entities = {'periods': [], 'dates': []}
    query_lower = query.lower()

    # Extract relative periods
    relative_match = re.search(r'\b(?:last|past)\s+(\d+)\s+(month|months|year|years)\b', query_lower)
    if relative_match:
        count = int(relative_match.group(1))
        unit = relative_match.group(2)

        end_date = datetime.now()
        if 'month' in unit:
            start_date = end_date - timedelta(days=count * 30)
        else:  # years
            start_date = end_date - timedelta(days=count * 365)

        entities['periods'].append({
            'type': 'relative',
            'start_date': start_date,
            'end_date': end_date,
            'description': f'last {count} {unit}'
        })

    # Extract month/year combinations
    month_year_match = re.search(r'\b(?:in\s+)?(\w+)\s+(\d{4})\b', query_lower)
    if month_year_match:
        month_str = month_year_match.group(1)
        year = int(month_year_match.group(2))

        try:
            date_obj = dateparser.parse(f"{month_str} {year}")
            if date_obj:
                # Get start and end of month
                start_date = date_obj.replace(day=1)
                if start_date.month == 12:
                    end_date = start_date.replace(year=start_date.year + 1, month=1)
                else:
                    end_date = start_date.replace(month=start_date.month + 1)

                entities['dates'].append({
                    'type': 'month_year',
                    'start_date': start_date,
                    'end_date': end_date,
                    'description': f'{month_str} {year}'
                })
        except:
            pass

    # >>> ADD THE NEW BLOCK HERE <<<
    # This new block handles patterns for a standalone year like "in 2020" or "for the year 2022"
    year_only_match = re.search(r'\b(?:in|for|during|year)\s+(\d{4})\b', query_lower)
    if year_only_match:
        year = int(year_only_match.group(1))
        try:
            start_date = datetime(year, 1, 1)
            # Set end_date to the very end of the last day of the year
            end_date = datetime(year, 12, 31, 23, 59, 59)

            entities['dates'].append({
                'type': 'year_only',
                'start_date': start_date,
                'end_date': end_date,
                'description': f'the year {year}'
            })
        except ValueError:
            # This will catch invalid years, though the regex makes it unlikely
            pass

    return entities

def filter_df_by_temporal(df: pd.DataFrame, temporal_entities: dict | None) -> pd.DataFrame:
    """Filter a measurement DF by OR-ing multiple time windows on DATETIME if provided."""
    if df.empty or temporal_entities is None or "DATETIME" not in df.columns:
        return df
    windows = temporal_entities.get("periods", []) + temporal_entities.get("dates", [])
    if not windows:
        return df
    mask = None
    for w in windows:
        m = (df["DATETIME"] >= pd.to_datetime(w["start_date"])) & (df["DATETIME"] < pd.to_datetime(w["end_date"]))
        mask = m if mask is None else (mask | m)
    return df.loc[mask].copy() if mask is not None else df


def detect_enhanced_intent(query: str) -> dict:
    """Enhanced intent detection with geographic and temporal awareness."""

    q = query.lower()

    # Geographic query patterns
    geo_patterns = [
    'near', 'closest', 'nearest', 'around', 'in the vicinity',
    # existing:
    'equator', 'arabian sea', 'indian ocean', 'region',
    # new:
    'pacific ocean', 'atlantic ocean', 'mediterranean sea',
    'red sea', 'south china sea', 'caribbean sea', 'gulf of mexico'
    ]


    # Temporal query patterns
    temporal_patterns = [
        'last', 'past', 'recent', 'since', 'during', 'in march',
        'months', 'years', '2023', 'time period'
    ]

    # BGC comparison patterns
    bgc_patterns = [
        'compare', 'comparison', 'versus', 'vs', 'difference',
        'oxygen', 'chlorophyll', 'nitrate', 'ph', 'bgc'
    ]

    # Location search patterns
    location_patterns = [
        'nearest', 'closest', 'distance', 'location', 'coordinates'
    ]

    enhanced_region_patterns = [
        'lme', 'large marine ecosystem', 'ecoregion', 'water mass',
        'biogeochemical province', 'upwelling', 'subtropical gyre',
        'coral triangle', 'mediterranean', 'caribbean', 'benguela'
    ]

    intent = {
        'primary_action': 'general_query',
        'modifiers': [],
        'entities': {}
    }

    # Determine primary intent
    if any(pattern in q for pattern in location_patterns):
        intent['primary_action'] = 'find_nearest_floats'

    elif any(pattern in q for pattern in bgc_patterns) and any(pattern in q for pattern in ['region', 'sea', 'ocean']):
        intent['primary_action'] = 'compare_regional_bgc'

    elif any(pattern in q for pattern in ['profile', 'vertical', 'depth']):
        intent['primary_action'] = 'show_profiles'

    elif any(pattern in q for pattern in ['trajectory', 'path', 'track', 'movement']):
        intent['primary_action'] = 'show_trajectories'

    # Add modifiers
    if any(pattern in q for pattern in enhanced_region_patterns):
        intent['primary_action'] = 'enhanced_regional_analysis'
        intent['modifiers'].append('enhanced_geographic_filter')
    if any(pattern in q for pattern in geo_patterns):
        intent['modifiers'].append('geographic_filter')

    if any(pattern in q for pattern in temporal_patterns):
        intent['modifiers'].append('temporal_filter')

    if any(pattern in q for pattern in bgc_patterns):
        intent['modifiers'].append('bgc_analysis')

    # Extract entities
    intent['entities']['geographic'] = extract_geographic_entities(query)
    intent['entities']['temporal'] = extract_temporal_entities(query)

    return intent

@st.cache_data(show_spinner="Searching for floats in specified region...")
def find_floats_in_region(_pool, geographic_entities: list, temporal_entities: list = None) -> pd.DataFrame:
    """Find floats within specified geographic and temporal constraints."""

    if not _pool or not geographic_entities:
        return pd.DataFrame()

    conn = _get_conn(_pool)
    if not conn:
        return pd.DataFrame()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            base_query = """
                SELECT DISTINCT p.platform_number, p.cycle_number, p.latitude, p.longitude,
                       p.datetime, p.profile_id,
                       array_agg(DISTINCT m.parameter_name) as available_parameters
                FROM profiles p
                LEFT JOIN measurements m ON p.profile_id = m.profile_id
                WHERE 1=1
            """

            params = []
            conditions = []

            # Add geographic conditions
            for geo_entity in geographic_entities:
                if geo_entity['type'] == 'equatorial':
                    lat_min, lat_max = geo_entity['lat_range']
                    conditions.append("p.latitude BETWEEN %s AND %s")
                    params.extend([lat_min, lat_max])

                elif geo_entity['type'] == 'region' and 'bounds' in geo_entity:
                    bounds = geo_entity['bounds']
                    conditions.append(
                        "p.latitude BETWEEN %s AND %s AND p.longitude BETWEEN %s AND %s"
                    )
                    params.extend([bounds['lat'][0], bounds['lat'][1],
                                 bounds['lon'][0], bounds['lon'][1]])

            # Add temporal conditions
            if temporal_entities:
                tw = temporal_entities.get('periods', []) + temporal_entities.get('dates', [])
                if tw:
                    temp_clauses = []
                    for temp_entity in tw:
                        temp_clauses.append("p.datetime BETWEEN %s AND %s")
                        params.extend([temp_entity['start_date'], temp_entity['end_date']])
                    conditions.append("(" + " OR ".join(temp_clauses) + ")")


            if conditions:
                base_query += " AND " + " AND ".join(conditions)

            base_query += """
                GROUP BY p.platform_number, p.cycle_number, p.latitude, p.longitude, p.datetime, p.profile_id
                ORDER BY p.datetime DESC
            """

            cur.execute(base_query, params)
            results = cur.fetchall()

            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame()

    except Exception as e:
        st.error(f"Error finding floats in region: {e}")
        return pd.DataFrame()
    finally:
        _put_conn(_pool, conn)

@st.cache_data(show_spinner="Finding nearest floats...")
def find_nearest_floats(_pool, target_lat: float, target_lon: float, max_distance_km: float = 500, limit: int = 10) -> pd.DataFrame:
    """Find nearest floats to a target location using PostGIS."""

    if not _pool:
        return pd.DataFrame()

    conn = _get_conn(_pool)
    if not conn:
        return pd.DataFrame()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Use PostGIS to calculate distances
            query = """
                SELECT p.platform_number, p.cycle_number, p.latitude, p.longitude, p.datetime,
                       p.profile_id,
                       ST_Distance(
                           ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                           p.location_point::geography
                       ) / 1000.0 as distance_km,
                       array_agg(DISTINCT m.parameter_name) as available_parameters
                FROM profiles p
                LEFT JOIN measurements m ON p.profile_id = m.profile_id
                WHERE ST_DWithin(
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                    p.location_point::geography,
                    %s * 1000
                )
                GROUP BY p.platform_number, p.cycle_number, p.latitude, p.longitude,
                         p.datetime, p.profile_id, p.location_point
                ORDER BY distance_km ASC
                LIMIT %s
            """

            cur.execute(query, [target_lon, target_lat, target_lon, target_lat, max_distance_km, limit])
            results = cur.fetchall()

            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame()

    except Exception as e:
        st.error(f"Error finding nearest floats: {e}")
        return pd.DataFrame()
    finally:
        _put_conn(_pool, conn)

@st.cache_data(show_spinner="Analyzing BGC parameters...")
def analyze_bgc_parameters_by_region(_pool, geographic_entities: list, temporal_entities: list = None) -> pd.DataFrame:
    """Analyze BGC parameter availability and statistics by region."""

    if not _pool:
        return pd.DataFrame()

    conn = _get_conn(_pool)
    if not conn:
        return pd.DataFrame()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # BGC parameters of interest
            bgc_params = ['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700',
                         'DOWNWELLING_PAR', 'CDOM', 'TURBIDITY']

            base_query = """
                SELECT p.platform_number, p.latitude, p.longitude, p.datetime,
                        m.parameter_name, m.pressure, m.value_adjusted,
                        CASE
                            WHEN p.latitude BETWEEN -5 AND 5 THEN 'Equatorial Band'
                            WHEN p.latitude BETWEEN 8 AND 28 AND p.longitude BETWEEN 50 AND 78 THEN 'Arabian Sea'
                            WHEN p.latitude BETWEEN -40 AND 30 AND p.longitude BETWEEN 20 AND 120 THEN 'Indian Ocean'
                            WHEN p.latitude BETWEEN -60 AND 65 AND (p.longitude >= 120 OR p.longitude <= -70) THEN 'Pacific Ocean'
                            WHEN p.latitude BETWEEN -60 AND 70 AND p.longitude BETWEEN -80 AND 20 THEN 'Atlantic Ocean'
                            WHEN p.latitude BETWEEN 30 AND 46 AND p.longitude BETWEEN -6 AND 36 THEN 'Mediterranean Sea'
                            WHEN p.latitude BETWEEN 12 AND 30 AND p.longitude BETWEEN 32 AND 43 THEN 'Red Sea'
                            WHEN p.latitude BETWEEN 0 AND 25 AND p.longitude BETWEEN 99 AND 125 THEN 'South China Sea'
                            WHEN p.latitude BETWEEN 9 AND 25 AND p.longitude BETWEEN -85 AND -60 THEN 'Caribbean Sea'
                            WHEN p.latitude BETWEEN 18 AND 31 AND p.longitude BETWEEN -98 AND -80 THEN 'Gulf of Mexico'
                            ELSE 'Other'
                        END AS region
                FROM profiles p
                JOIN measurements m ON p.profile_id = m.profile_id
                WHERE m.parameter_name = ANY(%s)
                    AND m.value_adjusted IS NOT NULL
            """


            params = [bgc_params]
            conditions = []

            # Add geographic filters
            if geographic_entities:
                for geo_entity in geographic_entities:
                    if geo_entity['type'] == 'equatorial':
                        lat_min, lat_max = geo_entity['lat_range']
                        conditions.append("p.latitude BETWEEN %s AND %s")
                        params.extend([lat_min, lat_max])
                    elif geo_entity['type'] == 'region' and 'bounds' in geo_entity:
                        bounds = geo_entity['bounds']
                        conditions.append(
                            "p.latitude BETWEEN %s AND %s AND p.longitude BETWEEN %s AND %s"
                        )
                        params.extend([bounds['lat'][0], bounds['lat'][1],
                                     bounds['lon'][0], bounds['lon'][1]])

            # Add temporal filters
            if temporal_entities:
                tw = temporal_entities.get('periods', []) + temporal_entities.get('dates', [])
                if tw:
                    temp_clauses = []
                    for temp_entity in tw:
                        temp_clauses.append("p.datetime BETWEEN %s AND %s")
                        params.extend([temp_entity['start_date'], temp_entity['end_date']])
                    conditions.append("(" + " OR ".join(temp_clauses) + ")")

            if conditions:
                base_query += " AND " + " AND ".join(conditions)

            base_query += " ORDER BY p.datetime DESC, m.parameter_name, m.pressure"

            cur.execute(base_query, params)
            results = cur.fetchall()

            return pd.DataFrame(results) if results else pd.DataFrame()

    except Exception as e:
        st.error(f"Error analyzing BGC parameters: {e}")
        return pd.DataFrame()
    finally:
        _put_conn(_pool, conn)

@st.cache_data(show_spinner="Querying enhanced region definitions...")
def get_enhanced_regions_from_db(_pool, region_types: list = None) -> pd.DataFrame:
    """Get enhanced ocean region definitions from database."""
    if not _pool:
        return pd.DataFrame()

    conn = _get_conn(_pool)
    if not conn:
        return pd.DataFrame()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            base_query = """
            SELECT region_name, region_type,
                   ST_AsText(boundary) as boundary_wkt,
                   depth_min, depth_max, characteristics, description
            FROM enhanced_ocean_regions
            """
            params = []

            if region_types:
                base_query += " WHERE region_type = ANY(%s)"
                params.append(region_types)

            base_query += " ORDER BY region_type, region_name"

            cur.execute(base_query, params)
            results = cur.fetchall()

            return pd.DataFrame(results) if results else pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching enhanced regions: {e}")
        return pd.DataFrame()
    finally:
        _put_conn(_pool, conn)

@st.cache_data(show_spinner="Finding floats in enhanced regions...")
def find_floats_in_enhanced_regions(_pool, region_names: list, temporal_entities: dict = None) -> pd.DataFrame:
    """Find floats within enhanced region definitions."""
    if not _pool or not region_names:
        return pd.DataFrame()

    conn = _get_conn(_pool)
    if not conn:
        return pd.DataFrame()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            base_query = """
            SELECT DISTINCT p.platform_number, p.cycle_number, p.latitude, p.longitude,
                   p.datetime, p.profile_id, er.region_name, er.region_type,
                   array_agg(DISTINCT m.parameter_name) as available_parameters
            FROM profiles p
            JOIN enhanced_ocean_regions er ON ST_Within(p.location_point, er.boundary)
            LEFT JOIN measurements m ON p.profile_id = m.profile_id
            WHERE er.region_name = ANY(%s)
            """
            params = [region_names]

            # Add temporal filters if provided
            if temporal_entities:
                time_windows = temporal_entities.get('periods', []) + temporal_entities.get('dates', [])
                if time_windows:
                    temp_clauses = []
                    for tw in time_windows:
                        temp_clauses.append("p.datetime BETWEEN %s AND %s")
                        params.extend([tw['start_date'], tw['end_date']])
                    base_query += " AND (" + " OR ".join(temp_clauses) + ")"

            base_query += """
            GROUP BY p.platform_number, p.cycle_number, p.latitude, p.longitude,
                     p.datetime, p.profile_id, er.region_name, er.region_type
            ORDER BY p.datetime DESC
            """

            cur.execute(base_query, params)
            results = cur.fetchall()

            return pd.DataFrame(results) if results else pd.DataFrame()

    except Exception as e:
        st.error(f"Error finding floats in enhanced regions: {e}")
        return pd.DataFrame()
    finally:
        _put_conn(_pool, conn)


def create_geographic_map(df: pd.DataFrame, query_context: str = "") -> str:
    if df.empty:
        return "No geographic data available for mapping."

    df_cols = {c.upper(): c for c in df.columns}
    lat_col = df_cols.get("LATITUDE")
    lon_col = df_cols.get("LONGITUDE")
    float_col = df_cols.get("PLATFORM_NUMBER") or df_cols.get("FLOAT_ID")

    if not lat_col or not lon_col:
        return "No LAT/LON columns found in the data."

    fig = plot_floats_on_globe(df, lat_col=lat_col, lon_col=lon_col, float_col=float_col)
    st.plotly_chart(fig, use_container_width=True)

    n_floats = df[float_col].nunique() if float_col else len(df)
    date_range = ""
    datetime_col = df_cols.get("DATETIME")
    if datetime_col and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        date_range = f"from {df[datetime_col].min():%Y-%m-%d} to {df[datetime_col].max():%Y-%m-%d}"

    summary = f"Geographic map showing {len(df)} profiles from {n_floats} floats {date_range}."
    return summary

def create_bgc_comparison_plot(df: pd.DataFrame, query_context: str = "") -> str:
    """Create comparison plots for BGC parameters across regions."""

    if df.empty:
        return "No BGC data available for comparison."

    # Group by parameter and region
    if 'parameter_name' not in df.columns or 'region' not in df.columns:
        return "Missing required columns for BGC comparison."

    # Create subplots for each parameter
    parameters = df['parameter_name'].unique()

    if len(parameters) == 0:
        return "No BGC parameters found in the data."

    cols = min(2, len(parameters))
    rows = (len(parameters) + 1) // 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=parameters,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    colors = px.colors.qualitative.Set1

    for i, param in enumerate(parameters):
        param_data = df[df['parameter_name'] == param]

        if not param_data.empty:
            row = (i // cols) + 1
            col = (i % cols) + 1

            # Create box plot for each region
            for j, region in enumerate(param_data['region'].unique()):
                region_data = param_data[param_data['region'] == region]

                fig.add_trace(
                    px.box(region_data, y='value_adjusted', name=region)['data'][0],
                    row=row, col=col
                )

            fig.update_xaxes(title_text="Region", row=row, col=col)
            fig.update_yaxes(title_text=f"{param} Value", row=row, col=col)

    fig.update_layout(
        height=300 * rows,
        title_text=f"BGC Parameter Comparison - {query_context}",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"bgc_comparison_{query_context}_{datetime.now().timestamp()}")

    # Summary statistics
    param_counts = df.groupby(['region', 'parameter_name']).size().reset_index(name='measurement_count')

    st.subheader("Parameter Availability by Region")
    st.dataframe(param_counts.pivot(index='region', columns='parameter_name', values='measurement_count').fillna(0))

    return f"BGC comparison showing {len(parameters)} parameters across {df['region'].nunique()} regions."

def process_enhanced_query(user_query: str, db_pool, embedding_model, vector_collection):
    """Process enhanced queries with geographic and temporal awareness."""

    if not user_query.strip():
        return

    # Use enhanced RAG to find relevant floats and extract intent
    float_ids, context_docs, intent_info = find_relevant_floats_via_enhanced_rag(
        user_query, embedding_model, vector_collection
    )

    if not float_ids:
        st.warning("No relevant floats found for your query.")
        return

    entities = (intent_info or {}).get('entities', {}) or {}
    primary_action = (intent_info or {}).get('primary_action', 'general_query')
    geo_entities = entities.get('geographic', {}) or {}
    temporal_entities = entities.get('temporal', {}) or {}


    # Execute appropriate action based on intent
    if primary_action == 'find_nearest_floats':
        # Extract coordinates if provided
        coords = geo_entities.get('coordinates', [])
        if coords:
            target_coord = coords[0]
            df = find_nearest_floats(
                db_pool, target_coord['lat'], target_coord['lon']
            )
            if not df.empty:
                st.header("🗺️ Nearest ARGO Floats")
                create_geographic_map(df, "Nearest floats search")
                st.dataframe(df)
        else:
            st.warning("Please provide coordinates for nearest float search (e.g., 'near 15.5N 65.2E')")

    elif primary_action == 'compare_regional_bgc':
        # Analyze BGC parameters by region
        df = analyze_bgc_parameters_by_region(db_pool, geo_entities.get('regions', []), temporal_entities)
        if not df.empty:
            st.header("🧪 BGC Parameter Analysis")
            create_bgc_comparison_plot(df, user_query)
        else:
            st.warning("No BGC data found for the specified region and time period.")

    elif primary_action == 'show_profiles':
    # Step A: find candidate profiles with geo/time filters
        df = find_floats_in_region(db_pool, geo_entities.get('regions', []), temporal_entities)
        if not df.empty:
            floats_key = tuple(df['platform_number'].unique())
            detailed_df = fetch_data_from_database(db_pool, floats_key, temporal_entities)
            st.header("📊 Profile Analysis")
            if 'regions' in geo_entities:
                create_geographic_map(df, "Profile locations")
            variables = ['TEMP', 'PSAL'] + [c for c in detailed_df.columns if c in ['DOXY', 'CHLA', 'NITRATE']]
            plot_profile(detailed_df, variables[:4])


            # Create profile plots
            variables = ['TEMP', 'PSAL'] + [col for col in detailed_df.columns
                                         if col in ['DOXY', 'CHLA', 'NITRATE']]
            plot_profile(detailed_df, variables[:4])  # Limit to 4 variables

            # Show data summary
            with st.expander("📋 Profile Summary"):
                st.dataframe(detailed_df.describe())
    elif primary_action == 'enhanced_regional_analysis':
        # Use enhanced region definitions
        enhanced_regions_df = get_enhanced_regions_from_db(db_pool)

        if not enhanced_regions_df.empty:
            # Match query to enhanced regions
            matched_regions = []
            query_lower = user_query.lower()

            for _, region in enhanced_regions_df.iterrows():
                if any(term in query_lower for term in [
                    region['region_name'].lower(),
                    region['region_type'].lower()
                ]):
                    matched_regions.append(region['region_name'])

            if matched_regions:
                df = find_floats_in_enhanced_regions(db_pool, matched_regions, temporal_entities)

                if not df.empty:
                    st.header("🌍 Enhanced Regional Analysis")
                    create_geographic_map(df, f"Enhanced regions: {', '.join(matched_regions)}")

                    # Show region characteristics
                    region_info = enhanced_regions_df[
                        enhanced_regions_df['region_name'].isin(matched_regions)
                    ]
                    st.subheader("Region Characteristics")
                    st.dataframe(region_info[['region_name', 'region_type', 'description']])
                else:
                    st.warning(f"No floats found in enhanced regions: {', '.join(matched_regions)}")
            else:
                st.warning("Could not match query to any enhanced region definitions.")
        else:
            st.warning("No enhanced region definitions found in database.")


    else:
        # Default: show trajectories with any filters applied
        df = find_floats_in_region(db_pool, geo_entities.get('regions', []), temporal_entities)
        if not df.empty:
            st.header("🗺️ Float Trajectories")
            create_geographic_map(df, user_query)
            plot_trajectory_summary = plot_trajectory(df)
            st.info(plot_trajectory_summary)


# --- Page and App Configuration ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="FloatChat | ARGO Data Explorer", layout="wide", page_icon="")

# --- UPDATED CSS THEME ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Hammersmith+One&display=swap');
    
    /* Light whitish background */
    .stApp { 
        font-family: 'Inter', sans-serif; 
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 50%, #f1f3f4 100%);
        min-height: 100vh; 
    }
    
    .main .block-container { 
        background: rgba(255, 255, 255, 0.98); 
        border-radius: 20px; 
        padding: 2rem; 
        backdrop-filter: blur(10px); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.08); 
        margin-top: 2rem; 
        margin-bottom: 2rem; 
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Enhanced Sidebar Styling to Match Main Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%) !important;
        border-right: 2px solid rgba(102, 126, 234, 0.2) !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar content styling */
    .css-1d391kg {
        background: transparent !important;
        padding: 1.5rem !important;
    }
    
    /* Sidebar text and headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .css-1lcbmhc {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .css-1lcbmhc {
        color: #333333 !important;
    }
    
    /* Sidebar widgets styling */
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Sidebar file uploader */
    [data-testid="stSidebar"] .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stFileUploader > div:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }
    
    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Sidebar checkbox */
    [data-testid="stSidebar"] .stCheckbox {
        background: rgba(255, 255, 255, 0.7) !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox > label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar info box */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #2c3e50 !important;
    }
    
    /* Sidebar success/error messages */
    [data-testid="stSidebar"] .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%) !important;
        border: 1px solid #4caf50 !important;
        color: #2e7d32 !important;
    }
    
    [data-testid="stSidebar"] .stError {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%) !important;
        border: 1px solid #f44336 !important;
        color: #c62828 !important;
    }
    
    [data-testid="stSidebar"] .stWarning {
        background: linear-gradient(135deg, #fff3e0 0%, #fef7e0 100%) !important;
        border: 1px solid #ff9800 !important;
        color: #ef6c00 !important;
    }
    
    /* Sidebar markdown styling */
    [data-testid="stSidebar"] .stMarkdown {
        color: #2c3e50 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(102, 126, 234, 0.3) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Updated header with Hammersmith One font */
    .main-header { 
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
        padding: 3rem 2rem; 
        border-radius: 20px; 
        margin-bottom: 2rem; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.08); 
        color: #2c3e50; 
        text-align: center; 
        position: relative; 
        overflow: hidden; 
        animation: slideInDown 0.8s ease-out; 
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Hammersmith One font for main header with black color */
    .main-header h1 {
        font-family: 'Hammersmith One', sans-serif !important;
        font-weight: 400 !important;
        font-size: 3.5rem !important;
        letter-spacing: 0.02em !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
        margin: 0 !important;
        color: #000000 !important;
        line-height: 1.2 !important;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        font-size: 1.2rem !important;
        color: #333333 !important;
        margin: 1rem 0 0 0 !important;
        letter-spacing: 0.3px !important;
    }
    
    .main-header::before { 
        content: ''; 
        position: absolute; 
        top: -50%; 
        left: -50%; 
        width: 200%; 
        height: 200%; 
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent); 
        animation: shine 3s infinite; 
    }
    
    @keyframes slideInDown { 
        from { transform: translateY(-50px); opacity: 0; } 
        to { transform: translateY(0); opacity: 1; } 
    }
    
    @keyframes shine { 
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); } 
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); } 
    }
    
    /* Light KPI cards */
    .kpi-card { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border-radius: 15px; 
        padding: 2rem; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        border: 1px solid rgba(0,0,0,0.08); 
        margin-bottom: 1.5rem; 
        transition: all 0.3s ease; 
        position: relative; 
        overflow: hidden; 
        animation: slideInUp 0.6s ease-out; 
    }
    
    .kpi-card:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 12px 30px rgba(0,0,0,0.12); 
    }
    
    .kpi-card::before { 
        content: ''; 
        position: absolute; 
        top: 0; 
        left: 0; 
        right: 0; 
        height: 4px; 
        background: linear-gradient(90deg, #e3f2fd, #f3e5f5); 
    }
    
    @keyframes slideInUp { 
        from { transform: translateY(30px); opacity: 0; } 
        to { transform: translateY(0); opacity: 1; } 
    }
    
    /* Light chart containers */
    .chart-container { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border-radius: 15px; 
        padding: 2rem; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        margin-bottom: 2rem; 
        border: 1px solid rgba(0,0,0,0.08); 
        transition: all 0.3s ease; 
        animation: fadeIn 0.8s ease-out; 
    }
    
    .chart-container:hover { 
        box-shadow: 0 8px 25px rgba(0,0,0,0.12); 
    }
    
    @keyframes fadeIn { 
        from { opacity: 0; transform: scale(0.98); } 
        to { opacity: 1; transform: scale(1); } 
    }
    
    /* Light themed buttons */
    .stButton > button { 
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
        color: #2c3e50; 
        border: 2px solid rgba(0,0,0,0.1); 
        border-radius: 12px; 
        padding: 0.8rem 2rem; 
        font-weight: 600; 
        font-size: 1rem; 
        transition: all 0.3s ease; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
        position: relative; 
        overflow: hidden; 
    }
    
    .stButton > button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 15px rgba(0,0,0,0.15); 
        background: linear-gradient(135deg, #bbdefb 0%, #e1bee7 100%);
    }
    
    /* Light themed tabs */
    .stTabs [data-baseweb="tab-list"] { 
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
        border-radius: 15px; 
        padding: 0.5rem; 
        box-shadow: 0 3px 15px rgba(0,0,0,0.08); 
        gap: 0.5rem; 
        border: 1px solid rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] { 
        background: transparent; 
        border-radius: 10px; 
        padding: 1rem 1.5rem; 
        font-weight: 600; 
        color: #2c3e50; 
        transition: all 0.3s ease; 
        border: 2px solid transparent; 
    }
    
    .stTabs [data-baseweb="tab"]:hover { 
        background: rgba(227, 242, 253, 0.5); 
        border-color: rgba(227, 242, 253, 0.8); 
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
        color: #1565c0; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
        border-color: rgba(21, 101, 192, 0.2);
    }
    
    /* Light chat container */
    .chat-container { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border-radius: 15px; 
        padding: 2rem; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        margin: 1rem 0; 
        border: 1px solid rgba(0,0,0,0.08); 
    }
    
    /* Light themed metrics */
    [data-testid="metric-container"] { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border: 1px solid rgba(0,0,0,0.08); 
        padding: 1rem; 
        border-radius: 12px; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.08); 
        transition: all 0.3s ease; 
    }
    
    [data-testid="metric-container"]:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 15px rgba(0,0,0,0.12); 
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- NLP model (cached) ---
@st.cache_resource(show_spinner="Loading NLP model...")
def get_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

nlp = get_nlp()

# --- Helper: Entity extraction ---
def extract_entities(query: str, aliases: dict):
    """
    Detect natural-language aliases in the query and map them to dataframe column names.
    Uses spaCy PhraseMatcher when available, otherwise falls back to substring matching.
    """
    if not aliases:
        return []

    query_lower = query.lower()

    # If spaCy is available, use PhraseMatcher for robust multi-word matching
    if nlp:
        doc = nlp(query_lower)
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(alias) for alias in aliases.keys()]
        matcher.add("ARGO_VARS", patterns)
        matches = matcher(doc)
        found = []
        for _match_id, start, end in matches:
            span = doc[start:end].text
            if span in aliases:
                found.append(aliases[span])
        return list(set(found))

    # Fallback: simple substring search
    found = []
    for alias, col in aliases.items():
        if alias in query_lower:
            found.append(col)
    return list(set(found))

# --- Helpers for argopy -> DataFrame processing ---
def _actual_name(ds, upper_name):
    """Return actual variable name in ds matching an uppercase key, or None."""
    for k in ds.variables:
        if k.upper() == upper_name:
            return k
    return None

def _prefer_adjusted(ds, raw_name):
    """Return adjusted var name if raw missing and adjusted exists, otherwise raw if exists, else None."""
    raw = _actual_name(ds, raw_name)
    adj = _actual_name(ds, f"{raw_name}_ADJUSTED")
    if raw is not None:
        return raw
    if adj is not None:
        return adj
    return None

@st.cache_data(show_spinner="Fetching ARGO data via Argopy...")
def fetch_argo_data_via_argopy(floats_key: tuple, dataset_type: str = "phy"):
    """
    Use argopy to fetch data for one or more WMO float IDs and return an xarray.Dataset.
    The input key must be hashable; pass a tuple of ints.
    dataset_type can be 'phy' or 'bgc'
    """
    float_ids = list(floats_key)
    try:
        if dataset_type == "bgc":
            # Set BGC dataset and fetch all BGC parameters
            ds = ArgoDataFetcher(ds='bgc').float(float_ids).to_xarray()
        else:
            # Default physical dataset
            ds = ArgoDataFetcher().float(float_ids).to_xarray()
        return ds
    except Exception as e:
        # If BGC fails, try with physical dataset as fallback
        if dataset_type == "bgc":
            st.warning(f"BGC data fetch failed, falling back to physical data: {e}")
            ds = ArgoDataFetcher().float(float_ids).to_xarray()
            return ds
        else:
            raise e

@st.cache_data(show_spinner="Processing dataset...")
def process_argopy_dataset(_ds: xr.Dataset, floats_key: tuple, dataset_type: str = "phy") -> pd.DataFrame:
    """
    Convert an argopy xarray Dataset to a combined DataFrame, with caching keyed
    by a hashable floats_key (e.g., tuple of WMO IDs). The Dataset itself is
    ignored for hashing by using an underscore-prefixed parameter name.
    """
    if _ds is None or len(_ds.variables) == 0:
        return pd.DataFrame()

    ds = _ds  # local alias

    # Core variables (always present)
    var_LAT = _actual_name(ds, "LATITUDE")
    var_LON = _actual_name(ds, "LONGITUDE")
    var_JULD = _actual_name(ds, "JULD")
    var_TIME = _actual_name(ds, "TIME")
    var_PRES = _prefer_adjusted(ds, "PRES")
    var_TEMP = _prefer_adjusted(ds, "TEMP")
    var_PSAL = _prefer_adjusted(ds, "PSAL")
    var_NPROF = _actual_name(ds, "N_PROF")
    var_PLAT = _actual_name(ds, "PLATFORM_NUMBER")
    var_CYCLE = _actual_name(ds, "CYCLE_NUMBER")

    # BGC variables (if dataset_type is 'bgc' or if variables are present)
    bgc_vars = {}
    bgc_var_names = [
        "DOXY",           # Dissolved oxygen
        "CHLA",           # Chlorophyll-a
        "NITRATE",        # Nitrate
        "BBP",            # Backscattering coefficient (multiple wavelengths possible)
        "BBP700",         # Backscattering at 700nm
        "BBP532",         # Backscattering at 532nm
        "PH_IN_SITU_TOTAL", # pH
        "DOWNWELLING_PAR",  # Photosynthetically Available Radiation
        "DOWN_IRRADIANCE380", # Downwelling irradiance 380nm
        "DOWN_IRRADIANCE412", # Downwelling irradiance 412nm
        "DOWN_IRRADIANCE490", # Downwelling irradiance 490nm
        "CDOM",           # Colored Dissolved Organic Matter
        "TURBIDITY",      # Turbidity
    ]

    for bgc_var in bgc_var_names:
        bgc_vars[bgc_var] = _prefer_adjusted(ds, bgc_var)

    # Collect all variables to extract
    vars_to_take = [
        v for v in [var_LAT, var_LON, var_JULD, var_TIME, var_PRES, var_TEMP, var_PSAL, var_NPROF, var_PLAT, var_CYCLE]
        if v is not None
    ]

    # Add available BGC variables
    for bgc_var, actual_var in bgc_vars.items():
        if actual_var is not None:
            vars_to_take.append(actual_var)

    if not vars_to_take:
        return pd.DataFrame()

    sub_ds = ds[vars_to_take]

    # Flatten to DataFrame
    df = sub_ds.to_dataframe().reset_index()
    df.columns = [str(c).upper() for c in df.columns]

    # Normalize measurement variable names (core variables)
    if "TEMP_ADJUSTED" in df.columns and "TEMP" not in df.columns:
        df["TEMP"] = df["TEMP_ADJUSTED"]
    if "PSAL_ADJUSTED" in df.columns and "PSAL" not in df.columns:
        df["PSAL"] = df["PSAL_ADJUSTED"]
    if "PRES_ADJUSTED" in df.columns and "PRES" not in df.columns:
        df["PRES"] = df["PRES_ADJUSTED"]

    # Normalize BGC variable names
    bgc_normalizations = {
        "DOXY_ADJUSTED": "DOXY",
        "CHLA_ADJUSTED": "CHLA",
        "NITRATE_ADJUSTED": "NITRATE",
        "BBP_ADJUSTED": "BBP",
        "BBP700_ADJUSTED": "BBP700",
        "BBP532_ADJUSTED": "BBP532",
        "PH_IN_SITU_TOTAL_ADJUSTED": "PH_IN_SITU_TOTAL",
        "DOWNWELLING_PAR_ADJUSTED": "DOWNWELLING_PAR",
        "DOWN_IRRADIANCE380_ADJUSTED": "DOWN_IRRADIANCE380",
        "DOWN_IRRADIANCE412_ADJUSTED": "DOWN_IRRADIANCE412",
        "DOWN_IRRADIANCE490_ADJUSTED": "DOWN_IRRADIANCE490",
    }

    for adjusted_name, normalized_name in bgc_normalizations.items():
        if adjusted_name in df.columns and normalized_name not in df.columns:
            df[normalized_name] = df[adjusted_name]

    # FLOAT_ID from PLATFORM_NUMBER if available
    if "PLATFORM_NUMBER" in df.columns:
        df["FLOAT_ID"] = df["PLATFORM_NUMBER"].astype(str).str.extract(r"(\d+)").fillna("unknown")
    else:
        df["FLOAT_ID"] = "unknown"

    # Convert numeric columns safely (core + BGC variables)
    numeric_cols = ["LATITUDE", "LONGITUDE", "JULD", "PRES", "TEMP", "PSAL",
                   "DOXY", "CHLA", "NITRATE", "BBP", "BBP700", "BBP532", "PH_IN_SITU_TOTAL",
                   "DOWNWELLING_PAR", "DOWN_IRRADIANCE380", "DOWN_IRRADIANCE412", "DOWN_IRRADIANCE490"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # DATETIME from TIME if present, else from JULD (days since 1950-01-01)
    if "TIME" in df.columns:
        try:
            df["DATETIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        except Exception:
            df["DATETIME"] = pd.NaT
    elif "JULD" in df.columns:
        try:
            df["DATETIME"] = pd.to_datetime("1950-01-01") + pd.to_timedelta(df["JULD"], unit="D")
        except Exception:
            try:
                converted = []
                for val in df["JULD"].values:
                    if pd.isna(val):
                        converted.append(pd.NaT)
                    else:
                        dt = cftime.num2date(val, "days since 1950-01-01")
                        converted.append(pd.to_datetime(str(dt)))
                df["DATETIME"] = pd.to_datetime(pd.Series(converted), errors="coerce")
            except Exception:
                pass

    # Keep rows that have at least one measurement (TEMP, PSAL, PRES, or any BGC variable)
    measurement_cols = []
    for c in ["TEMP", "PSAL", "PRES", "DOXY", "CHLA", "NITRATE", "BBP", "BBP700", "BBP532", "PH_IN_SITU_TOTAL"]:
        if c in df.columns:
            measurement_cols.append(c)

    if measurement_cols:
        df = df.dropna(subset=measurement_cols, how="all")

    # Basic coordinate sanity checks: drop records without lat/lon
    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"], how="any")

    # Ensure DATETIME is pandas datetime
    if "DATETIME" in df.columns:
        try:
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        except Exception:
            pass

    return df.reset_index(drop=True)

# --- ARGO alias helpers and intent detection ---
def get_argo_aliases(all_columns):
    base_aliases = {
        # Core variables
        "salinity": "PSAL",
        "temperature": "TEMP",
        "pressure": "PRES",
        "depth": "PRES",
        "latitude": "LATITUDE",
        "longitude": "LONGITUDE",
        "time": "DATETIME",
        "date": "DATETIME",
        "julian day": "JULD",
        "location": "LATITUDE",
        "position": "LONGITUDE",

        # BGC variables
        "oxygen": "DOXY",
        "dissolved oxygen": "DOXY",
        "o2": "DOXY",
        "chlorophyll": "CHLA",
        "chlorophyll a": "CHLA",
        "chla": "CHLA",
        "chl": "CHLA",
        "chl a": "CHLA",
        "nitrate": "NITRATE",
        "no3": "NITRATE",
        "backscatter": "BBP",
        "backscattering": "BBP",
        "bbp": "BBP",
        "optical backscatter": "BBP",
        "ph": "PH_IN_SITU_TOTAL",
        "acidity": "PH_IN_SITU_TOTAL",
        "par": "DOWNWELLING_PAR",
        "irradiance": "DOWNWELLING_PAR",
        "photosynthetically active radiation": "DOWNWELLING_PAR",
        "light": "DOWNWELLING_PAR",
        "turbidity": "TURBIDITY",
        "particles": "BBP",
        "suspended particles": "BBP",
    }
    return {alias: col for alias, col in base_aliases.items() if col in all_columns}

def detect_intent(query: str):
    q = (query or "").lower()
    if any(kw in q for kw in ["trajectory", "path", "map", "locations", "positions"]):
        return "plot_trajectory"
    if any(kw in q for kw in ["profile", "depth plot", "vertical profile", "vs depth"]):
        return "plot_profile"
    if any(kw in q for kw in ["time series", "over time", "trend", "timeseries", "time-series"]):
        return "plot_time_series"
    if any(kw in q for kw in ["summary", "describe", "statistics", "info", "overview"]):
        return "get_summary"
    return "general_query"

# --- Gemini integration with normalization & sanitization (guardrails) ---
def normalize_action(action):
    mapping = {
        "dataset_summary": "get_summary",
        "summary_dataset": "get_summary",
        "trajectory_plot": "plot_trajectory",
        "profile_plot": "plot_profile",
        "timeseries": "plot_time_series",
    }
    return mapping.get(action, action)

def sanitize_plan(plan: dict, df_columns: list):
    """
    Validate & sanitize the JSON plan returned by Gemini:
    - Ensure action is in supported set
    - Ensure variables refer to actual df columns
    """
    supported_actions = {"plot_trajectory", "plot_profile", "plot_time_series", "get_summary", "general_query"}

    if not plan:
        return {"action": "general_query", "entities": {"variables": [], "variable": None}, "comment": "No plan"}

    action = plan.get("action", "general_query")
    if action not in supported_actions:
        action = "general_query"

    entities = plan.get("entities", {}) or {}

    valid_vars = []
    if "variables" in entities and isinstance(entities["variables"], (list, tuple)):
        for v in entities["variables"]:
            if isinstance(v, str) and v.upper() in df_columns:
                valid_vars.append(v.upper())

    var = entities.get("variable")
    if isinstance(var, str) and var.upper() in df_columns:
        var_norm = var.upper()
    else:
        var_norm = None

    return {"action": action, "entities": {"variables": valid_vars, "variable": var_norm}, "comment": plan.get("comment", "")}

def get_llm_plan(query: str, df_columns: list, gemini_model, rag_context: list = None):
    """
    Send query to Gemini to get a plan, parse JSON and sanitize result.
    If Gemini is not available or response unparseable, return None to trigger fallback.
    """
    if not gemini_model:
        st.sidebar.error("LLM model not initialized.")
        return None

    system_prompt = f"""
You are an expert AI assistant for oceanography data analysis. Your task is to act as a query planner.
Convert the user's natural language query into a structured JSON command.
Output ONLY the JSON plan. Do not include any other text or explanations.

The available data columns are: {df_columns}
The available actions are: "plot_trajectory", "plot_profile", "plot_time_series", "get_summary", "data_qa".

Here are some examples of how to convert a query to a plan:

---
User Query: "Show me the path of the floats."
JSON Plan:
{{
  "action": "plot_trajectory",
  "entities": {{
    "variables": [],
    "variable": null
  }},
  "comment": "User wants to see the geographic locations of the floats, so I will plot the trajectory map."
}}
---
User Query: "Plot vertical profiles for temperature and oxygen."
JSON Plan:
{{
  "action": "plot_profile",
  "entities": {{
    "variables": ["TEMP", "DOXY"],
    "variable": null
  }},
  "comment": "User requested vertical profiles for temperature and oxygen, so I will generate a profile plot for TEMP and DOXY."
}}
---
User Query: "What is the trend of chlorophyll over time?"
JSON Plan:
{{
  "action": "plot_time_series",
  "entities": {{
    "variables": [],
    "variable": "CHLA"
  }},
  "comment": "User is asking for a trend over time, so I will create a time-series plot for chlorophyll (CHLA)."
}}
---
User Query: "List the unique float IDs in the dataset."
JSON Plan:
{{
  "action": "data_qa",
  "entities": {{}},
  "comment": "User is asking a direct question about the data's contents, so I will use the data question-answering tool."
}}
---

Important: BGC variables include oxygen (DOXY), chlorophyll (CHLA), nitrate (NITRATE), pH (PH_IN_SITU_TOTAL), backscattering (BBP), and irradiance parameters.
Ensure that any variables you select are present in the provided list of available columns.
"""
    full_prompt = f'{system_prompt}\n\nUser Query: "{query}"'

    try:
        # Generate content using Gemini, requesting a JSON response
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        response_text = response.text
        plan = json.loads(response_text)

        # Sanitize and return the plan
        plan["action"] = normalize_action(plan.get("action", "general_query"))
        if "action" not in plan or "entities" not in plan:
            raise ValueError("Invalid plan structure returned by LLM")

        return sanitize_plan(plan, df_columns)

    except Exception as e:
        st.sidebar.error(f"LLM planning error: {e}")
        # Attempt to recover JSON from a potentially malformed response
        try:
            # Check if the error or response text contains a JSON object
            error_text = str(e)
            if hasattr(e, 'last_response') and e.last_response:
                 error_text = e.last_response.text
            elif 'text' in locals() and response_text:
                error_text = response_text

            m = re.search(r"(\{.*\})", error_text, flags=re.DOTALL)
            if m:
                plan = json.loads(m.group(1))
                return sanitize_plan(plan, df_columns)
        except Exception:
            return None # Return None to trigger fallback if everything fails
        return None

def get_llm_data_analysis(user_query: str, df: pd.DataFrame, plot_type: str, gemini_model, variables: list = None):
    """
    Prompts Gemini to perform a natural language analysis of a pandas DataFrame.
    """
    if not gemini_model:
        return "The LLM AI analyst is not available."

    if df.empty:
        return "The analysis could not be performed as no data was available."

    # Create a concise, contextual summary of the data for the LLM
    data_summary = create_contextual_summary(df, plot_type, variables)

    variable_context = ""
    if variables:
        variable_context = f"The analysis should focus specifically on these variables: {', '.join(variables)}."

    # Add units to help the model be more precise
    units = {"TEMP": "Celsius", "PSAL": "PSU", "PRES": "dbar", "DOXY": "micromole/kg", "CHLA": "mg/m^3", "NITRATE": "micromole/kg"}
    units_context = f"Assume the following standard units for analysis: {units}"

    # Construct a detailed and constrained prompt for Gemini
    system_prompt = f"""
You are an expert AI oceanographer. Your task is to analyze the provided data summary and provide an insightful interpretation in natural language.
The user's original query was: "{user_query}"
A "{plot_type}" visualization has already been generated from the data.
{variable_context}
{units_context}

--- DATA SUMMARY ---
{data_summary}
--- END DATA SUMMARY ---

**Instructions:**
1.  **Grounding:** Base your entire analysis *exclusively* on the data in the summary above. Do not invent information or make assumptions.
2.  **Structure:** Structure your response using markdown with a main heading `## Data Analysis` and relevant subheadings (`###`).
3.  **Content:**
    - Start with a brief overview of the dataset's characteristics based on the summary.
    - Analyze the key features you observe (e.g., for a profile, discuss stratification, thermocline, halocline, chlorophyll maximums by referencing the depth-layer stats; for a time series, discuss the significance of the trend).
    - Provide a concluding interpretation that directly answers the user's original question.
4.  **Clarity:** Do not just repeat numbers. Explain what they *mean* in an oceanographic context. For example, instead of saying "The surface temperature is 25 and the deep temperature is 5," say "The data shows strong thermal stratification, with a 20-degree difference between the warm surface layer and the cold deep layer, indicating a pronounced thermocline."
"""
    try:
        # Generate content with Gemini
        response = gemini_model.generate_content(system_prompt)
        return response.text

    except Exception as e:
        st.sidebar.error(f"LLM analysis error: {e}")
        return "There was an error connecting to the LLM AI analyst."

import io
import sys
import pandas as pd
from psycopg2.extras import RealDictCursor

# Place this function before get_llm_data_analysis
def create_contextual_summary(df: pd.DataFrame, plot_type: str, variables: list = None):
    """Creates a tailored, context-aware data summary for the LLM."""
    if df.empty:
        return "No data available."

    summary_parts = []
    # Always include the overall description
    summary_parts.append(f"Overall Statistical Overview:\n{df.describe().to_string()}")

    if plot_type == "Vertical Profile" and 'PRES' in df.columns:
        summary_parts.append("\n--- Depth-Layer Analysis ---")
        depth_bins = {
            "Surface (0-50 dbar)": (0, 50),
            "Thermocline Layer (50-200 dbar)": (50, 200),
            "Deep Layer (>500 dbar)": (500, np.inf)
        }
        for name, (d_min, d_max) in depth_bins.items():
            subset = df[(df['PRES'] >= d_min) & (df['PRES'] < d_max)]
            if not subset.empty and variables:
                summary_parts.append(f"\nStatistics for {name}:\n{subset[variables].describe().to_string()}")

    elif plot_type == "Time Series" and 'DATETIME' in df.columns and variables:
        summary_parts.append("\n--- Trend Analysis ---")
        variable = variables[0]
        try:
            # Drop NaNs before processing
            temp_df = df[["DATETIME", variable]].dropna()
            if len(temp_df) < 2:
                raise ValueError("Not enough data points for trend analysis.")

            x = temp_df["DATETIME"].map(pd.Timestamp.toordinal).values
            y = temp_df[variable].values

            if len(x) >= 5 and np.isfinite(y).all():
                slope, _, _, p_value, _ = linregress(x, y)
                trend_desc = "statistically significant" if p_value < 0.05 else "not statistically significant"
                summary_parts.append(f"Linear trend for {variable}: {slope:.6f} units/day (p-value: {p_value:.4f}, which is {trend_desc}).")
        except Exception as e:
            summary_parts.append(f"Could not compute a linear trend for {variable}: {e}")

    return "\n".join(summary_parts)

# --- Context helpers for summaries (adds user query into text without changing core logic) ---
def _query_context_prefix():
    q = st.session_state.get("last_query", "")
    return f"" if q else ""

# --- Visualization functions (return NLP summary strings) ---
def plot_trajectory(df: pd.DataFrame):
    """
    Plot map of float trajectories and return a plain-language summary that
    also provides brief oceanographic interpretation and references the user's query.
    """
    if df.empty:
        st.warning("No data available to plot trajectories.")
        return "No trajectory data available."

    unique_floats = df["FLOAT_ID"].unique() if "FLOAT_ID" in df.columns else []
    truncated = False
    if len(unique_floats) > 50:
        st.info(f"Plotting first 50 of {len(unique_floats)} floats to preserve performance.")
        selected_ids = unique_floats[:50]
        plot_df = df[df["FLOAT_ID"].isin(selected_ids)].copy()
        truncated = True
    else:
        plot_df = df.copy()

    hover_cols = {}
    # Include BGC variables in hover data
    for col in ["DATETIME", "TEMP", "PSAL", "DOXY", "CHLA", "NITRATE", "PH_IN_SITU_TOTAL"]:
        if col in plot_df.columns:
            if ptypes.is_numeric_dtype(plot_df[col]):
                hover_cols[col] = ":.2f"
            else:
                hover_cols[col] = True

    fig = px.scatter_mapbox(
        plot_df,
        lat="LATITUDE",
        lon="LONGITUDE",
        color="FLOAT_ID",
        hover_name="FLOAT_ID",
        hover_data=hover_cols,
        title="ARGO Float Trajectories",
        mapbox_style="open-street-map",
        zoom=2,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"trajectory_{datetime.now().timestamp()}")

    n_floats = plot_df["FLOAT_ID"].nunique() if "FLOAT_ID" in plot_df.columns else 0
    time_span = ""
    if "DATETIME" in plot_df.columns and plot_df["DATETIME"].notna().any():
        try:
            tmin = plot_df["DATETIME"].min()
            tmax = plot_df["DATETIME"].max()
            time_span = f" Data spans {tmin.date()} to {tmax.date()}."
        except Exception:
            time_span = ""

    # Simple geographic extent and interpretation
    extent_txt = ""
    try:
        lat_min, lat_max = float(plot_df["LATITUDE"].min()), float(plot_df["LATITUDE"].max())
        lon_min, lon_max = float(plot_df["LONGITUDE"].min()), float(plot_df["LONGITUDE"].max())
        extent_txt = f" Geographic extent ~ lat {lat_min:.1f}° to {lat_max:.1f}°, lon {lon_min:.1f}° to {lon_max:.1f}°."
    except Exception:
        pass

    # Basic interpretation: more spread -> stronger advection/eddy field implied
    spread_comment = ""
    try:
        lat_spread = float(plot_df["LATITUDE"].max() - plot_df["LATITUDE"].min())
        lon_spread = float(plot_df["LONGITUDE"].max() - plot_df["LONGITUDE"].min())
        if lat_spread > 20 or lon_spread > 40:
            spread_comment = " The wide spatial spread suggests influence of strong currents and mesoscale eddies typical of open-ocean regimes."
        elif lat_spread < 5 and lon_spread < 10:
            spread_comment = " The compact cluster implies more localized dynamics or short deployment duration."
    except Exception:
        pass

    # BGC data availability context
    bgc_context = ""
    bgc_vars_present = [v for v in ["DOXY", "CHLA", "NITRATE", "PH_IN_SITU_TOTAL", "BBP"] if v in plot_df.columns]
    if bgc_vars_present:
        bgc_context = f" Biogeochemical data available for: {', '.join(bgc_vars_present)}."

    ctx = _query_context_prefix()
    summary = f"{ctx}Plotted trajectories of {n_floats} float(s).{time_span}{extent_txt}{spread_comment}{bgc_context}"
    if truncated:
        summary += " Only the first 50 floats were plotted to preserve performance."
    summary += " Hover points to inspect per-float measurements where available."
    return summary

def plot_profile(df: pd.DataFrame, variables: list):
    """
    Plot vertical profile(s) and return a summary with oceanographic interpretation
    (e.g., stratification, haline gradients, BGC gradients) and an explicit link back to the query.
    """
    if df.empty:
        st.warning("No data available to plot profiles.")
        return "No profile data available."

    if not variables:
        st.warning("No variables specified for profile plot.")
        return "No variables specified to plot profiles."

    if "PRES" not in df.columns:
        st.error("No 'PRES' (pressure) column in dataframe — cannot plot depth profiles.")
        return "Missing PRES (pressure) column; cannot create depth profiles."

    available = [v for v in variables if v in df.columns]
    missing = [v for v in variables if v not in df.columns]
    if missing:
        st.warning(f"These variables are missing and will be skipped: {', '.join(missing)}")
    if not available:
        st.error("None of the requested variables are available for plotting.")
        return "Requested profile variables are not available in the dataset."

    columns_needed = ["FLOAT_ID", "PRES"] + available
    plot_df = df[columns_needed].melt(
        id_vars=["FLOAT_ID", "PRES"], value_vars=available, var_name="VARIABLE", value_name="VALUE"
    ).dropna(subset=["VALUE"])

    if plot_df.empty:
        st.warning("No valid measurements found to plot.")
        return "No valid measurements found for requested profile variables."

    fig = px.line(
        plot_df,
        x="VALUE",
        y="PRES",
        color="VARIABLE",
        line_group="FLOAT_ID",
        hover_name="FLOAT_ID",
        title=f"Profiles: {', '.join(available)} vs Depth",
    )
    fig.update_yaxes(autorange="reversed", title_text="Depth (Pressure in dbar)")
    st.plotly_chart(fig, use_container_width=True, key=f"profile_{'_'.join(available)}_{datetime.now().timestamp()}")

    # Enhanced quantitative ranges and interpretation for BGC variables
    parts = []
    for v in available:
        col = df[v].dropna()
        if not col.empty and ptypes.is_numeric_dtype(col):
            rng = f"{col.min():.2f}–{col.max():.2f}"
        else:
            rng = "n/a"

        interp_bits = []
        try:
            surf = df.loc[df["PRES"] <= 10, v].dropna()
            deep = df.loc[df["PRES"] >= 1000, v].dropna()
            if not surf.empty and not deep.empty:
                s_med, d_med = float(surf.median()), float(deep.median())
                diff = s_med - d_med

                # Core variable interpretations
                if v == "TEMP":
                    if diff > 2.0:
                        interp_bits.append("warm surface layer relative to depth, indicating thermal stratification")
                    elif diff < -0.5:
                        interp_bits.append("surface cooler than depth, which is atypical and may reflect mixing or fronts")
                elif v == "PSAL":
                    if diff < -0.05:
                        interp_bits.append("fresher surface layer consistent with rainfall/river influence")
                    elif diff > 0.05:
                        interp_bits.append("saltier surface layer suggesting evaporation dominance")

                # BGC variable interpretations
                elif v == "DOXY":
                    if diff > 50:  # Higher oxygen at surface
                        interp_bits.append("oxygen-rich surface waters indicating photosynthetic production and air-sea exchange")
                    elif diff < -20:  # Lower oxygen at surface
                        interp_bits.append("surface oxygen depletion suggesting strong biological consumption or upwelling of oxygen-poor waters")
                elif v == "CHLA":
                    if s_med > d_med * 2:  # Much higher chlorophyll at surface
                        interp_bits.append("elevated surface chlorophyll indicating phytoplankton blooms in the euphotic zone")
                    # Look for deep chlorophyll maximum
                    if len(df.loc[(df["PRES"] >= 50) & (df["PRES"] <= 150), v].dropna()) > 5:
                        dcm_layer = df.loc[(df["PRES"] >= 50) & (df["PRES"] <= 150), v].dropna()
                        if not dcm_layer.empty and dcm_layer.max() > s_med * 1.5:
                            interp_bits.append("deep chlorophyll maximum detected in subsurface layers")
                elif v == "NITRATE":
                    if diff < -2:  # Lower nitrate at surface
                        interp_bits.append("surface nitrate depletion indicating nutrient uptake by phytoplankton")
                    if s_med < 1 and d_med > 10:
                        interp_bits.append("typical oligotrophic profile with nitrate-depleted surface and nutrient-rich deep waters")
                elif v == "PH_IN_SITU_TOTAL":
                    if diff > 0.1:  # Higher pH at surface
                        interp_bits.append("elevated surface pH consistent with CO2 uptake by photosynthesis")
                    elif diff < -0.05:  # Lower pH at surface
                        interp_bits.append("reduced surface pH potentially indicating ocean acidification or CO2 outgassing")

            # Gradient analysis for upper water column
            upper = df.loc[(df["PRES"] >= 0) & (df["PRES"] <= 200), ["PRES", v]].dropna()
            if len(upper) >= 5:
                up_sorted = upper.sort_values("PRES")
                x = up_sorted["PRES"].values
                y = up_sorted[v].values
                if np.isfinite(y).all():
                    slope, _, _, _, _ = linregress(x, y)
                    if v == "TEMP" and slope < -0.01:
                        interp_bits.append("a pronounced thermocline (temperature decreasing with depth) in the upper 200 dbar")
                    elif v == "PSAL" and abs(slope) > 0.005:
                        interp_bits.append("a near-surface haline gradient (halocline signal)")
                    elif v == "DOXY" and slope < -0.5:
                        interp_bits.append("oxygen declining with depth, typical of the oxycline")
                    elif v == "NITRATE" and slope > 0.05:
                        interp_bits.append("nitrate increasing with depth, showing the nitracline")
        except Exception:
            pass

        interp_txt = f" ({'; '.join(interp_bits)})" if interp_bits else ""
        parts.append(f"{v} {interp_txt}")

    ctx = _query_context_prefix()
    summary = f"{ctx}Plotted vertical profiles for variables: {', '.join(available)}. " + " ; ".join(parts) + ". Pressure increases downward on the y-axis."
    return summary

def plot_time_series(df: pd.DataFrame, variable: str):
    """
    Plot a variable over time and return a summary that includes:
    - linear trend with significance,
    - seasonality hints,
    - BGC-specific interpretations,
    - explicit linkage to the user's query.
    """
    if df.empty:
        st.warning("No data available to plot time series.")
        return "No time-series data available."

    if variable is None:
        st.error("No variable specified for time series.")
        return "No variable specified for the time-series plot."

    if variable not in df.columns:
        st.error(f"Requested variable '{variable}' not present in data.")
        return f"Variable '{variable}' not available in dataset."

    if "DATETIME" not in df.columns or df["DATETIME"].isna().all():
        st.error("DATETIME column missing or invalid — cannot plot time series.")
        return "Missing DATETIME column; time-series cannot be plotted."

    plot_df = df.sort_values("DATETIME").dropna(subset=[variable])
    if plot_df.empty:
        st.warning("No valid time-series data found for the requested variable.")
        return f"No valid time-series measurements found for {variable}."

    fig = px.line(plot_df, x="DATETIME", y=variable, color="FLOAT_ID", title=f"Time Series: {variable} over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{variable}_{datetime.now().timestamp()}")

    # Statistical analysis
    slope = None
    p_value = None
    try:
        x = plot_df["DATETIME"].map(pd.Timestamp.toordinal).values
        y = plot_df[variable].values
        if len(x) >= 2 and np.isfinite(y).all():
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
    except Exception:
        slope = None

    seasonal_var = None
    monthly_means = None
    try:
        monthly_var = plot_df.groupby(plot_df["DATETIME"].dt.month)[variable].var()
        if not monthly_var.empty:
            seasonal_var = monthly_var.mean()
        monthly_means = plot_df.groupby(plot_df["DATETIME"].dt.month)[variable].mean()
    except Exception:
        seasonal_var = None

    tmin = plot_df["DATETIME"].min()
    tmax = plot_df["DATETIME"].max()
    ctx = _query_context_prefix()
    summary = f"{ctx}{variable} time series plotted for {plot_df['FLOAT_ID'].nunique()} float(s). Date range: {tmin.date()} to {tmax.date()}."

    # Trend analysis
    if slope is not None:
        summary += f" Linear trend ≈ {slope:.6f} {variable}/day"
        if p_value is not None:
            if p_value < 0.05:
                trend_dir = "increasing" if slope > 0 else "decreasing"
                summary += f" (statistically significant, {trend_dir}, p={p_value:.3f})."

                # BGC-specific trend interpretations
                if variable == "DOXY" and slope < -0.01:
                    summary += " This declining oxygen trend could indicate deoxygenation processes."
                elif variable == "CHLA" and slope > 0.001:
                    summary += " Increasing chlorophyll may reflect enhanced productivity or bloom events."
                elif variable == "PH_IN_SITU_TOTAL" and slope < -0.0001:
                    summary += " Declining pH suggests ocean acidification processes."
                elif variable == "NITRATE" and slope < -0.01:
                    summary += " Decreasing nitrate may indicate enhanced nutrient utilization."
            else:
                summary += f" (not statistically significant, p={p_value:.3f})."
        else:
            summary += "."
    else:
        summary += " A robust linear trend could not be computed (insufficient or non-numeric data)."

    # Seasonal analysis
    if seasonal_var is not None and not np.isnan(seasonal_var):
        summary += f" Average seasonal (monthly) variance ≈ {seasonal_var:.6f}."
        try:
            if monthly_means is not None and len(monthly_means.dropna()) >= 3:
                max_m = int(monthly_means.idxmax())
                min_m = int(monthly_means.idxmin())
                summary += f" Monthly means peak in {datetime(2000, max_m, 1).strftime('%B')} and are lowest in {datetime(2000, min_m, 1).strftime('%B')}."

                # BGC-specific seasonal interpretations
                if variable == "CHLA":
                    if max_m in [3,  4, 5] or max_m in [9, 10, 11]:  # Spring/Fall
                        summary += " This seasonal pattern is typical of temperate phytoplankton bloom cycles."
                elif variable == "DOXY":
                    if max_m in [12, 1, 2]:  # Winter
                        summary += " Higher winter oxygen is consistent with enhanced solubility in cold waters and reduced biological consumption."
        except Exception:
            pass
    else:
        summary += " Seasonal variance could not be computed."

    # --- ANOMALY DETECTION AND AI EXPLANATION ---
    if slope is not None and not plot_df.empty:
        mean_val = plot_df[variable].mean()
        std_val = plot_df[variable].std()
        # Find points more than 2.5 standard deviations from the mean
        anomalies_df = plot_df[np.abs(plot_df[variable] - mean_val) > (2.5 * std_val)]

        if not anomalies_df.empty:
            st.warning(f"Found {len(anomalies_df)} potential anomalies in the time series for {variable}.")
            with st.expander("View Anomalous Data Points"):
                st.dataframe(anomalies_df)

            anomalies_summary = anomalies_df[['DATETIME', variable]].to_string()
            prompt = f"""
You are an expert AI oceanographer. The following data points in a time series for {variable} were identified as statistical anomalies:

{anomalies_summary}

Based on oceanographic principles, please provide a few potential explanations for these anomalies. Consider possibilities such as:
- Mesoscale features (e.g., eddies, fronts)
- Biological events (e.g., phytoplankton blooms impacting BGC variables)
- Atmospheric events (e.g., storms causing mixing)
- Instrument malfunction or biofouling

Provide a concise, bulleted list of plausible causes.
"""
            if gemini_model:
                with st.spinner("AI is interpreting the anomalies..."):
                    ai_anomaly_explanation = gemini_model.generate_content(prompt).text
                    st.info("AI Interpretation of Anomalies:")
                    st.markdown(ai_anomaly_explanation)
            else:
                st.info("LLM model not available for anomaly interpretation.")

    return summary

# --- Main App UI & Flow ---
st.set_page_config(page_title="FloatChat | ARGO Data Explorer", layout="wide", page_icon="")
st.title("FloatChat: ARGO Data Explorer")
st.markdown("A conversational interface for ARGO ocean data.")

# --- Gemini API Configuration ---
@st.cache_resource
def configure_gemini():
    """Configure the Gemini API with a key from Streamlit secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("LLM_API_KEY secret not found. Please add it to your secrets.toml")
        return None
    try:
        genai.configure(api_key=api_key)
        # Using a model that supports JSON output format
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model
    except Exception as e:
        st.error(f"Failed to configure LLM: {e}")
        return None

# --- Initialize all components ---
db_pool = init_database_pool()
chroma_client = init_chromadb_client()
vector_collection = get_vector_collection(chroma_client)
embedding_model = load_embedding_model()
gemini_model = configure_gemini() # Initialize Gemini model

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_query" not in st.session_state: st.session_state.last_query = ""

# --- MODIFIED: CORRECTED INITIALIZATION ---
# Initialize the main DataFrame in session state so it persists across reruns
if "main_df" not in st.session_state:
    st.session_state.main_df = pd.DataFrame()

# df is now just a convenient alias for the persistent DataFrame
df = st.session_state.main_df

# ==============================================================================
# --- MAIN APP LOGIC based on data_source selection ---
# ==============================================================================

# --- SIDEBAR FOR DATA SELECTION AND SETUP ---

with st.sidebar:
    st.header("Database")
    if db_pool:
        st.success("PostgreSQL connected")
        if st.button("Update Schema"):
            with st.spinner("Applying schema…"):
                ok = create_database_schema(db_pool)
                if ok:
                    st.success("Schema up-to-date")
                else:
                    st.error("Schema failed!")
    else:
        st.error("DB connection unavailable")

    st.header("AI Model")
    if gemini_model:
        st.success("LLM API connected")
    else:
        st.error("LLM connection failed")


with st.sidebar:
    st.header("1. Data Source Selection")
    data_source = st.radio(
        "Choose how to select data:",
        ("Smart Search (RAG)", "Live Argopy Fetch", "PostgreSQL Database"),
        index=0,
        help="Use RAG for natural language search. Use Live Fetch/DB for manual float selection."
    )

    # Initialize variables
    float_ids = []
    dataset_type = "phy"

# --- Conditional UI based on Data Source ---
    if data_source == "Smart Search (RAG)":
        st.info("For Smart Search, type your query in the main chat window to find and visualize relevant data automatically.")
        st.subheader("Example Prompts by Region")

    # Use the specific list of regions provided by the user
        defined_regions = [
            "Indian Ocean",
            "Benguela Current LME",
            "Atlantic Ocean",
            "Subtropical Gyre - North Pacific",
            "Atlantic Ocean Equatorial Band",
            "Caribbean Sea",
            "Coral Triangle",
            "Equatorial Band",
            "Mediterranean Sea",
            "North Atlantic Deep Water",
            "Persian Gulf",
            "South China Sea"
        ]

    # Use an expander to keep the sidebar clean
        with st.expander("Select a predefined region to query"):
            for region in defined_regions:
                # Create a button for each region
                if st.button(region, key=f"region_{region}"):
                # When a button is clicked, update the main text area's content
                    st.session_state.rag_query = f"Analyze temperature and salinity profiles in the {region}"
                # Rerun the script to immediately show the updated text in the box
                    st.rerun()
            else:
                st.caption("No predefined regions found. Ensure the database schema has been created.")

    else: # UI for "Live Argopy Fetch" and "PostgreSQL Database"
        if data_source == "Live Argopy Fetch":
            st.header("2. Dataset Type")
            dataset_type = st.radio("Choose dataset type:", ("phy", "bgc"),
                                    format_func=lambda x: "Physical" if x == "phy" else "BGC")
        else:
            st.header("2. Dataset Type")
            st.caption("Database mode loads all available parameters.")


        st.header("3. Float Selection")
        def _parse_float_ids(text):
            ids = re.split(r'[,\s]+', (text or "").strip())
            return [int(i) for i in ids if i.isdigit()]

        if data_source == "PostgreSQL Database":
            if db_pool:
                available_db_floats = get_available_floats_from_db(db_pool)
                if available_db_floats:
                    selected_db_floats = st.multiselect(
                        "Select from available floats in the database:",
                        options=available_db_floats,
                        default=available_db_floats[:2] if len(available_db_floats) > 1 else available_db_floats
                    )
                    float_ids = selected_db_floats
                else:
                    st.warning("No floats found in the database. Upload data via 'Live Argopy Fetch' first.")
            else:
                st.error("Database connection unavailable.")
        else:  # Live Argopy Fetch
            default_floats = "6902746" if dataset_type == "phy" else "5906439, 1901393"
            float_ids_text = st.text_input("WMO float IDs (comma-separated)", value=default_floats)
            with st.expander("Or use Argovis Selection Tool"):
                components.iframe("https://argovis.colorado.edu/argo", height=600)
                pasted_wmos = st.text_area("Paste WMO IDs from selection tool here")
            float_ids = _parse_float_ids(pasted_wmos) if pasted_wmos else _parse_float_ids(float_ids_text)

# --- ENHANCED RAG WORKFLOW ---
if data_source == "Smart Search (RAG)":
    st.header("Ask a Question to Find & Visualize Data")

    rag_query = st.text_area(
        "Enter your question here to automatically find relevant floats and generate visualizations",
        key="rag_query",
        placeholder="e.g., Show me salinity profiles near the equator in March 2023"
    )

    if st.button("🔍 Ask AI Assistant", key="rag_button"):
        if rag_query:
            st.session_state.chat_history.append({"role": "user", "content": rag_query})
            st.session_state.last_query = rag_query

            with st.spinner("Performing enhanced semantic search..."):
                float_ids, rag_context, intent_info = find_relevant_floats_via_enhanced_rag(
                    rag_query, embedding_model, vector_collection
                )

            if float_ids:
                floats_key = tuple(sorted(list(set(map(str, float_ids)))))
                with st.spinner(f"Fetching data for {len(floats_key)} relevant float(s) from database..."):
                    safe_temporal = ((intent_info or {}).get('entities', {}) or {}).get('temporal')
                    # --- MODIFIED: Assign to session_state.main_df ---
                    st.session_state.main_df = fetch_data_from_database(db_pool, floats_key, safe_temporal)
                    df = st.session_state.main_df # Update the local alias

                if not df.empty:
                    # --- NEW: Show geographic context map first ---
                    st.header("📍 Geographic Context for Your Query")
                    map_summary = create_geographic_map(df, rag_query)
                    st.caption(map_summary)
                    st.markdown("---") # Add a separator

                    all_columns = df.columns.tolist()
                    with st.spinner("AI is planning the best visualization..."):
                        plan = get_llm_plan(rag_query, all_columns, gemini_model, rag_context)

                    if plan and "action" in plan:
                        action = plan.get("action")
                        entities = plan.get("entities", {})
                        comment = plan.get("comment", f"Executing action: {action}")

                        llm_analysis_text = ""
                        variables_for_analysis = []

                        if action == "plot_trajectory":
                            st.header("Analysis: Float Trajectory")
                            plot_trajectory(df) # Just creates the plot
                            with st.spinner("AI Oceanographer is analyzing the data..."):
                                llm_analysis_text = get_llm_data_analysis(rag_query, df, "Trajectory Map", gemini_model)

                        elif action == "plot_profile":
                            st.header("Analysis: Vertical Profiles")
                            variables = entities.get("variables", [])
                            if not variables:
                                variables = [v for v in ["TEMP", "PSAL", "DOXY", "CHLA"] if v in all_columns][:2]
                            variables_for_analysis = variables
                            plot_profile(df, variables) # Just creates the plot
                            with st.spinner("AI Oceanographer is analyzing the data..."):
                                llm_analysis_text = get_llm_data_analysis(rag_query, df, "Vertical Profile", gemini_model, variables_for_analysis)

                        elif action == "plot_time_series":
                            st.header("Analysis: Time Series Plot")
                            variable = entities.get("variable")
                            if not variable:
                                variable = next((v for v in ["TEMP", "DOXY", "CHLA"] if v in all_columns), None)
                            variables_for_analysis = [variable] if variable else []
                            plot_time_series(df, variable) # Just creates the plot
                            with st.spinner("AI Oceanographer is analyzing the data..."):
                                llm_analysis_text = get_llm_data_analysis(rag_query, df, "Time Series", gemini_model, variables_for_analysis)

                        elif action == "get_summary":
                            st.header("Analysis: Data Summary")
                            st.dataframe(df.describe())
                            llm_analysis_text = "Displayed the raw descriptive statistics for the retrieved data."

                        if llm_analysis_text:
                            st.markdown(llm_analysis_text)
                            # Combine comment and analysis into one message
                            combined_response = f"**Action:** {action.replace('_', ' ').title()}\n\n{llm_analysis_text}"
                            st.session_state.chat_history.append({"role": "assistant", "content": combined_response})
                            # --- MODIFIED: Save the last analysis text to session state ---
                            st.session_state.last_analysis = llm_analysis_text

                        with st.expander("Show Retrieved Data"): st.dataframe(df)
                        with st.expander("Show RAG Context"): st.json(rag_context)
                    else:
                        st.warning("The AI could not create a valid visualization plan. Displaying raw data and context.")
                        st.dataframe(df)
                        with st.expander("Show RAG Context"): st.json(rag_context)
                else:
                    msg = f"Found {len(float_ids)} relevant floats, but could not retrieve their detailed data from PostgreSQL."
                    st.error(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
            else:
                msg = "I could not find any floats in the database that match your query."
                st.warning(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
        else:
            st.warning("Please enter a question before clicking the button.")

else:  # --- Logic for "Live Argopy Fetch" and "PostgreSQL Database" modes ---
    floats_key = tuple(sorted(list(set(map(str, float_ids))))) if data_source == "PostgreSQL Database" else tuple(sorted(list(set(float_ids))))

    if floats_key:
        if data_source == "Live Argopy Fetch":
            st.info("📡 Data source: Live fetch from Argopy")
            ds = fetch_argo_data_via_argopy(floats_key, dataset_type)
            # --- MODIFIED: Assign to session_state.main_df ---
            st.session_state.main_df = process_argopy_dataset(ds, floats_key, dataset_type)
        elif data_source == "PostgreSQL Database":
            st.info("🗄️ Data source: Querying PostgreSQL Database")
            if not db_pool:
                st.error("Database connection is unavailable. Cannot fetch data.")
                st.stop()
            # --- MODIFIED: Assign to session_state.main_df ---
            st.session_state.main_df = fetch_data_from_database(db_pool, floats_key, None)

        df = st.session_state.main_df # Update the local alias

    if not df.empty:
        # --- Option to upload live data ---
        if data_source == "Live Argopy Fetch" and db_pool:
            if st.button("⬆️ Upload this dataset to PostgreSQL", type="primary"):
                with st.spinner("Saving data…"):
                    ok = store_data_in_database(df, db_pool)
                    st.success("Upload complete!" if ok else "Upload failed!")
                    st.cache_data.clear()
                    st.rerun()

        with st.sidebar:
            st.header("4. Analyze a Subset")
            present_ids = sorted(df["FLOAT_ID"].dropna().unique().tolist())
            if present_ids:
                selected_ids = st.multiselect("Analyze a subset of the loaded floats", options=present_ids, default=present_ids)
                if selected_ids and set(selected_ids) != set(present_ids):
                    df = df[df["FLOAT_ID"].isin(selected_ids)].copy()

        # --- Display loaded data info ---
        all_columns = df.columns.tolist()
        with st.expander("Click to see loaded data and available variables", expanded=False):
            st.write(f"Data shape: {df.shape}")
            st.dataframe(df.tail())

        st.header("📍 Default View: Float Trajectories")
        plot_trajectory(df)

        # --- Chat UI for pre-loaded data ---
        st.header("Chat with Loaded Data")
        user_query_viz = st.text_area("Your question:", key="user_query_viz",
                                  placeholder="e.g., Plot oxygen and chlorophyll profiles for the loaded floats")
        if st.button("🔎 Send Query", key="viz_button"):
            st.session_state.chat_history.append({"role": "user", "content": user_query_viz})
            st.session_state.last_query = user_query_viz

            with st.spinner("AI is planning the analysis..."):
                plan = get_llm_plan(user_query_viz, all_columns, gemini_model)

            # --- Execute Plan ---
            if plan and "action" in plan:
                action = plan.get("action")
                entities = plan.get("entities", {})
                comment = plan.get("comment", f"Executing action: {action}")

                llm_analysis_text = ""
                variables_for_analysis = []

                if action == "plot_trajectory":
                    st.header("Analysis: Float Trajectory")
                    plot_trajectory(df)
                    with st.spinner("AI Oceanographer is analyzing the data..."):
                        llm_analysis_text = get_llm_data_analysis(user_query_viz, df, "Trajectory Map", gemini_model)

                elif action == "plot_profile":
                    st.header("Analysis: Vertical Profiles")
                    variables = entities.get("variables", [])
                    if not variables:
                        variables = [v for v in ["TEMP", "PSAL", "DOXY", "CHLA"] if v in all_columns][:2]
                    variables_for_analysis = variables
                    plot_profile(df, variables)
                    with st.spinner("AI Oceanographer is analyzing the data..."):
                        llm_analysis_text = get_llm_data_analysis(user_query_viz, df, "Vertical Profile", gemini_model, variables_for_analysis)

                elif action == "plot_time_series":
                    st.header("Analysis: Time Series Plot")
                    variable = entities.get("variable")
                    if not variable:
                        variable = next((v for v in ["TEMP", "DOXY", "CHLA"] if v in all_columns), None)
                    variables_for_analysis = [variable] if variable else []
                    plot_time_series(df, variable)
                    with st.spinner("🔎 Oceanographer is analyzing the data..."):
                        llm_analysis_text = get_llm_data_analysis(user_query_viz, df, "Time Series", gemini_model, variables_for_analysis)

                elif action == "get_summary":
                    st.header("Analysis: Data Summary")
                    st.dataframe(df.describe())
                    llm_analysis_text = "Displayed the raw descriptive statistics for the retrieved data."

                # Combine comment and analysis, then display and save to history
                if llm_analysis_text:
                    st.markdown(llm_analysis_text)
                    combined_response = f"**Action:** {action.replace('_', ' ').title()}\n\n{llm_analysis_text}"
                    st.session_state.chat_history.append({"role": "assistant", "content": combined_response})
                    # --- MODIFIED: Save the last analysis text to session state ---
                    st.session_state.last_analysis = llm_analysis_text

            else:
                # Enhanced fallback if Gemini plan fails
                comment = "AI analysis failed. Attempting a default visualization based on your query."
                st.warning(comment)
                fallback_intent = detect_intent(user_query_viz)
                fallback_entities = extract_entities(user_query_viz, get_argo_aliases(all_columns))
                llm_analysis_text = ""
                action_title = "Fallback Trajectory"

                if fallback_intent == "plot_profile":
                    action_title = "Fallback Vertical Profiles"
                    st.header(action_title)
                    vars_to_plot = fallback_entities if fallback_entities else [v for v in ["TEMP", "PSAL"] if v in all_columns]
                    plot_profile(df, vars_to_plot)
                    with st.spinner("AI Oceanographer is analyzing the fallback data..."):
                        llm_analysis_text = get_llm_data_analysis(user_query_viz, df, "Vertical Profile", gemini_model, vars_to_plot)

                elif fallback_intent == "plot_time_series":
                    action_title = "Fallback Time Series"
                    st.header(action_title)
                    var_to_plot = fallback_entities[0] if fallback_entities else next((v for v in ["TEMP", "PSAL"] if v in all_columns), None)
                    plot_time_series(df, var_to_plot)
                    with st.spinner("AI Oceanographer is analyzing the fallback data..."):
                        llm_analysis_text = get_llm_data_analysis(user_query_viz, df, "Time Series", gemini_model, [var_to_plot])

                else: # Default to trajectory
                    st.header(action_title)
                    plot_trajectory(df)
                    with st.spinner("AI Oceanographer is analyzing the fallback data..."):
                        llm_analysis_text = get_llm_data_analysis(user_query_viz, df, "Trajectory Map", gemini_model)

                # Combine fallback comment and analysis, then display and save
                if llm_analysis_text:
                    st.markdown(llm_analysis_text)
                    combined_response = f"**Action:** {action_title}\n\n{comment}\n\n{llm_analysis_text}"
                    st.session_state.chat_history.append({"role": "assistant", "content": combined_response})
                    # --- MODIFIED: Save the last analysis text to session state ---
                    st.session_state.last_analysis = llm_analysis_text

    else:
        if data_source != "Smart Search (RAG)":
             st.info("Select a data source and float IDs in the sidebar to begin.")

# --- NEW: COMPARATIVE ANALYSIS UI ---
st.header("Comparative Analysis")
st.caption("Load a dataset using the controls above, then save it to a slot. Repeat for a second dataset to compare.")
col1, col2 = st.columns(2)

with col1:
    df_exists = not st.session_state.main_df.empty
    if st.button("Save Current Data to Slot 1", disabled=not df_exists):
        st.session_state.slot1_df = st.session_state.main_df.copy()
        st.session_state.slot1_query = st.session_state.get('last_query', 'N/A')
        # --- MODIFIED: Save the last analysis to the slot ---
        st.session_state.slot1_analysis = st.session_state.get('last_analysis', 'No analysis was generated for this data.')
        st.success(f"Data ({len(st.session_state.main_df)} rows) and its analysis saved to Slot 1.")
        st.rerun()

with col2:
    df_exists = not st.session_state.main_df.empty
    if st.button("Save Current Data to Slot 2", disabled=not df_exists):
        st.session_state.slot2_df = st.session_state.main_df.copy()
        st.session_state.slot2_query = st.session_state.get('last_query', 'N/A')
        # --- MODIFIED: Save the last analysis to the slot ---
        st.session_state.slot2_analysis = st.session_state.get('last_analysis', 'No analysis was generated for this data.')
        st.success(f"Data ({len(st.session_state.main_df)} rows) and its analysis saved to Slot 2.")
        st.rerun()

# --- Display comparison ---
slot1_df = st.session_state.get('slot1_df')
slot2_df = st.session_state.get('slot2_df')

if slot1_df is not None and slot2_df is not None:
    st.subheader("Side-by-Side Comparison")
    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"**Slot 1:** `{st.session_state.get('slot1_query', 'N/A')}`")
        if not slot1_df.empty:
            with st.expander("View Saved AI Analysis for Slot 1"):
                st.markdown(st.session_state.get('slot1_analysis', 'N/A'))
        else:
            st.warning("Slot 1 data is empty.")

    with colB:
        st.markdown(f"**Slot 2:** `{st.session_state.get('slot2_query', 'N/A')}`")
        if not slot2_df.empty:
            with st.expander("View Saved AI Analysis for Slot 2"):
                st.markdown(st.session_state.get('slot2_analysis', 'N/A'))
        else:
            st.warning("Slot 2 data is empty.")

    if st.button("Generate AI Comparison"):
        with st.spinner("AI is comparing the saved analyses..."):
            # --- MODIFIED: Use saved analyses instead of generating new summaries ---
            analysis1 = st.session_state.get('slot1_analysis', 'No analysis available.')
            analysis2 = st.session_state.get('slot2_analysis', 'No analysis available.')

            comparison_prompt = f"""
            You are an expert AI oceanographer. Your task is to compare and contrast the two oceanographic analyses provided below in a tabular format.
            Do not analyze the raw data; instead, synthesize the key findings from the existing analyses.

            --- ANALYSIS FOR DATASET 1 ---
            Original Query: "{st.session_state.get('slot1_query', 'N/A')}"
            {analysis1}
            --- END OF ANALYSIS 1 ---

            --- ANALYSIS FOR DATASET 2 ---
            Original Query: "{st.session_state.get('slot2_query', 'N/A')}"
            {analysis2}
            --- END OF ANALYSIS 2 ---

            **Instructions:**
            1.  **Synthesize Findings:** Compare the conclusions from Analysis 1 and Analysis 2.
            2.  **Identify Key Differences:** Focus on the main differences highlighted in the analyses, such as thermal stratification, halocline structure, chlorophyll maximums, or nutrient profiles.
            3.  **Structure:** Provide a detailed comparison in markdown format with a clear heading.
            4.  **Clarity:** Explain the significance of the differences. For example, "Dataset 1 shows a much stronger thermocline, suggesting it is from a region with more intense surface heating than Dataset 2."
            """

            if gemini_model:
                comparison_analysis = gemini_model.generate_content(comparison_prompt).text
                st.markdown("### AI-Generated Comparison of Analyses")
                st.markdown(comparison_analysis)
            else:
                st.error("LLM model not available for comparison.")
elif slot1_df is not None or slot2_df is not None:
    st.info("One slot is filled. Save data to the second slot to enable comparison.")


# --- Display Chat History (works for all modes) ---
# **MODIFICATION**: This entire block is updated for clarity with explicit labels.
if st.session_state.chat_history:
    st.markdown("---")
    st.header("Conversation History & Export")
    st.caption("Select conversations below to include them in your PDF export.")

    # This will hold the conversations selected by the user
    selected_chats_for_pdf = []

    # Iterate through the history in pairs (user query + assistant response)
    for i in range(0, len(st.session_state.chat_history), 2):
        user_message = st.session_state.chat_history[i]

        if i + 1 < len(st.session_state.chat_history):
            assistant_message = st.session_state.chat_history[i+1]

            # Create a concise title for the checkbox and expander
            title = user_message['content']
            if len(title) > 75:
                title = title[:72] + "..."

            # --- SELECTION UI ---
            # Create a checkbox for each conversation pair
            if st.checkbox(f"**Select:** {title}", key=f"select_{i}"):
                # If checked, add the pair to our list for PDF generation
                selected_chats_for_pdf.append({
                    'user': user_message,
                    'assistant': assistant_message
                })

            # --- DISPLAY UI ---
            # Keep the expander to view the details
            with st.expander("View Conversation Details"):
                with st.chat_message("user"):
                    st.markdown("**Your Query:**")
                    st.markdown(user_message["content"])

                with st.chat_message("assistant"):
                    st.markdown("**AI Assistant's Response:**")
                    st.markdown(assistant_message["content"])

    st.markdown("---")

    # --- DOWNLOAD BUTTON LOGIC ---
    # Generate PDF bytes from the list of selected chats
    pdf_bytes = create_pdf_from_history(selected_chats_for_pdf)

    # The button is disabled if the list is empty (nothing selected)
    st.download_button(
        label="📄 Download Selected as PDF",
        data=pdf_bytes,
        file_name=f"FloatChat_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        disabled=not selected_chats_for_pdf,
        help="Select one or more conversations above to enable download."
    )