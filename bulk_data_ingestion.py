# bulk_data_ingestion.py - Automated Data Ingestion Script for Enhanced FloatChat

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
import argopy
from argopy import DataFetcher as ArgoDataFetcher
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime, timedelta
import json
import xarray as xr
import cftime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArgoDataIngestionPipeline:
    """
    Automated data ingestion pipeline for ARGO float data
    Supports bulk ingestion, BGC data, and automatic embedding generation
    """
    
    def __init__(self, db_config, chroma_path="./chroma_argo_db"):
        self.db_config = db_config
        self.chroma_path = chroma_path
        self.db_pool = None
        self.chroma_client = None
        self.vector_collection = None
        self.embedding_model = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize database connections and models"""
        try:
            # Initialize PostgreSQL connection pool
            self.db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **self.db_config)
            logger.info("Database connection pool initialized")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.vector_collection = self.chroma_client.get_or_create_collection(
                name="argo_profiles_metadata",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def get_indian_ocean_floats(self, start_date="2020-01-01", end_date="2025-09-17"):
        """
        Get list of ARGO floats in Indian Ocean region for the specified time period
        Returns list of WMO float IDs
        """
        try:
            # Indian Ocean bounding box (rough approximation)
            # Longitude: 20°E to 120°E, Latitude: 30°S to 30°N
            lon_min, lon_max = 20, 120
            lat_min, lat_max = -30, 30
            
            logger.info(f"Searching for floats in Indian Ocean ({lat_min}°-{lat_max}°N, {lon_min}°-{lon_max}°E)")
            logger.info(f"Date range: {start_date} to {end_date}")
            
            # Use argopy to search for floats in the region
            # Note: This is a simplified approach - in production you might want to use
            # the Argo fleet monitoring tools or Argovis API for more precise selection
            
            region_fetcher = ArgoDataFetcher().region([lon_min, lon_max, lat_min, lat_max, 0, 2000, start_date, end_date])
            
            # Get float index to extract unique float IDs
            try:
                index_data = region_fetcher.to_index()
                if hasattr(index_data, 'platform_number'):
                    float_ids = index_data['platform_number'].unique().tolist()
                    logger.info(f"Found {len(float_ids)} unique floats in Indian Ocean region")
                    return [int(fid) for fid in float_ids if str(fid).isdigit()]
                else:
                    logger.warning("No platform_number found in index data")
                    return []
            except Exception as e:
                logger.warning(f"Failed to get region index, using fallback method: {e}")
                # Fallback: use a predefined list of known Indian Ocean floats
                return self._get_fallback_indian_ocean_floats()
                
        except Exception as e:
            logger.error(f"Failed to get Indian Ocean floats: {e}")
            return self._get_fallback_indian_ocean_floats()
    
    def _get_fallback_indian_ocean_floats(self):
        """Fallback list of known Indian Ocean ARGO floats"""
        # This is a curated list of floats known to operate in Indian Ocean
        # In production, this would be dynamically updated from Argo fleet monitoring
        fallback_floats = [
            2902211, 5906213, 6902746, 4902303, 2901725, 2902898, 1901133, 5906439, 6903270, 7900593, 6901768, 6901770, 2902306, 1901393, 2902340, 2901287, 2903290, 4902324, 4903340, 4903329, 4903369, 2902159, 2902899, 5905101, 5905103, 1901383, 1902231, 4902484, 5906622, 4902488, 5906214, 5903240, 6902901, 5904857, 6902904, 6901552, 2901202, 1902220, 2902690, 1902223, 6903099, 6903102, 1902222, 5906470
        ]
        logger.info(f"Using fallback list of {len(fallback_floats)} Indian Ocean floats")
        return fallback_floats
    
    def fetch_float_data(self, float_id, dataset_type='phy', max_retries=3):
        """
        Fetch data for a single float with retry logic
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {dataset_type} data for float {float_id} (attempt {attempt + 1})")
                
                if dataset_type == 'bgc':
                    fetcher = ArgoDataFetcher(ds='bgc')
                else:
                    fetcher = ArgoDataFetcher(ds='phy')
                
                ds = fetcher.float([float_id]).to_xarray()
                
                if ds is not None and len(ds.variables) > 0:
                    logger.info(f"Successfully fetched data for float {float_id}")
                    return ds
                else:
                    logger.warning(f"Empty dataset for float {float_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for float {float_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    logger.error(f"All attempts failed for float {float_id}")
                    return None
        
        return None
    
    def process_float_dataset(self, ds, float_id, include_bgc=False):
        """
        Process xarray dataset for a single float into DataFrame
        Enhanced version of the processing function from the main app
        """
        if ds is None or len(ds.variables) == 0:
            return pd.DataFrame()
        
        def _actual_name(ds, upper_name):
            for k in ds.variables:
                if k.upper() == upper_name:
                    return k
            return None
        
        def _prefer_adjusted(ds, raw_name):
            raw = _actual_name(ds, raw_name)
            adj = _actual_name(ds, f"{raw_name}_ADJUSTED")
            return adj if adj is not None else raw
        
        # Core variables
        var_mapping = {
            'LAT': _actual_name(ds, "LATITUDE"),
            'LON': _actual_name(ds, "LONGITUDE"),
            'JULD': _actual_name(ds, "JULD"),
            'TIME': _actual_name(ds, "TIME"),
            'PRES': _prefer_adjusted(ds, "PRES"),
            'TEMP': _prefer_adjusted(ds, "TEMP"),
            'PSAL': _prefer_adjusted(ds, "PSAL"),
            'PLAT': _actual_name(ds, "PLATFORM_NUMBER"),
            'CYCLE': _actual_name(ds, "CYCLE_NUMBER")
        }
        
        # BGC variables
        if include_bgc:
            bgc_params = ['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700', 
                         'DOWNWELLING_PAR', 'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE490']
            for param in bgc_params:
                var_mapping[param] = _prefer_adjusted(ds, param)
        
        # Extract available variables
        vars_to_take = [v for v in var_mapping.values() if v is not None]
        
        if not vars_to_take:
            logger.warning(f"No variables found for float {float_id}")
            return pd.DataFrame()
        
        try:
            sub_ds = ds[vars_to_take]
            df = sub_ds.to_dataframe().reset_index()
            df.columns = [str(c).upper() for c in df.columns]
            
            # Normalize column names
            if "TEMP_ADJUSTED" in df.columns and "TEMP" not in df.columns:
                df["TEMP"] = df["TEMP_ADJUSTED"]
            if "PSAL_ADJUSTED" in df.columns and "PSAL" not in df.columns:
                df["PSAL"] = df["PSAL_ADJUSTED"]
            if "PRES_ADJUSTED" in df.columns and "PRES" not in df.columns:
                df["PRES"] = df["PRES_ADJUSTED"]
            
            # Process BGC adjusted variables
            if include_bgc:
                bgc_params = ['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700', 
                             'DOWNWELLING_PAR', 'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE490']
                for param in bgc_params:
                    adj_col = f"{param}_ADJUSTED"
                    if adj_col in df.columns and param not in df.columns:
                        df[param] = df[adj_col]
            
            # Set float ID
            df["FLOAT_ID"] = str(float_id)
            if "PLATFORM_NUMBER" in df.columns:
                df["PLATFORM_NUMBER"] = df["PLATFORM_NUMBER"].astype(str)
            else:
                df["PLATFORM_NUMBER"] = str(float_id)
            
            # Convert numeric columns
            numeric_cols = ["LATITUDE", "LONGITUDE", "JULD", "PRES", "TEMP", "PSAL"]
            if include_bgc:
                numeric_cols.extend(['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700', 
                                   'DOWNWELLING_PAR', 'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE490'])
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Process datetime
            if "TIME" in df.columns:
                try:
                    df["DATETIME"] = pd.to_datetime(df["TIME"], errors="coerce")
                except:
                    df["DATETIME"] = pd.NaT
            elif "JULD" in df.columns:
                try:
                    df["DATETIME"] = pd.to_datetime("1950-01-01") + pd.to_timedelta(df["JULD"], unit="D")
                except:
                    df["DATETIME"] = pd.NaT
            
            # Data quality filters
            measurement_cols = [c for c in ["TEMP", "PSAL", "PRES"] if c in df.columns]
            if include_bgc:
                measurement_cols.extend([c for c in ['DOXY', 'CHLA', 'NITRATE'] if c in df.columns])
            
            if measurement_cols:
                df = df.dropna(subset=measurement_cols, how="all")
            
            # Geographic validation
            if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
                df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
                df = df[(df["LATITUDE"] >= -90) & (df["LATITUDE"] <= 90)]
                df = df[(df["LONGITUDE"] >= -180) & (df["LONGITUDE"] <= 180)]
            
            logger.info(f"Processed {len(df)} records for float {float_id}")
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error processing dataset for float {float_id}: {e}")
            return pd.DataFrame()
    
    def store_float_data(self, df, float_id):
        """Store processed float data in PostgreSQL"""
        if df.empty:
            return False
        
        try:
            conn = self.db_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                # Group by profile (platform + cycle)
                profiles = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER']).first().reset_index()
                
                for _, profile in profiles.iterrows():
                    # Insert/update profile
                    cursor.execute("""
                        INSERT INTO profiles (platform_number, cycle_number, latitude, longitude, datetime, juld, position_qc)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (platform_number, cycle_number) DO UPDATE SET
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        datetime = EXCLUDED.datetime,
                        juld = EXCLUDED.juld,
                        position_qc = EXCLUDED.position_qc
                        RETURNING profile_id
                    """, (
                        profile.get('PLATFORM_NUMBER'),
                        profile.get('CYCLE_NUMBER'),
                        profile.get('LATITUDE'),
                        profile.get('LONGITUDE'),
                        profile.get('DATETIME'),
                        profile.get('JULD'),
                        profile.get('POSITION_QC', 1)
                    ))
                    
                    result = cursor.fetchone()
                    profile_id = result['profile_id'] if result else None
                    
                    if profile_id:
                        # Get measurements for this profile
                        profile_data = df[
                            (df['PLATFORM_NUMBER'] == profile.get('PLATFORM_NUMBER')) &
                            (df['CYCLE_NUMBER'] == profile.get('CYCLE_NUMBER'))
                        ]
                        
                        # Store measurements
                        param_columns = ['TEMP', 'PSAL', 'PRES', 'DOXY', 'CHLA', 'NITRATE', 
                                       'PH_IN_SITU_TOTAL', 'BBP700', 'DOWNWELLING_PAR',
                                       'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE490']
                        
                        for _, row in profile_data.iterrows():
                            for param in param_columns:
                                if param in df.columns and pd.notna(row.get(param)):
                                    cursor.execute("""
                                        INSERT INTO measurements (profile_id, parameter_name, pressure, value, value_adjusted, qc_flag, data_mode)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                        ON CONFLICT DO NOTHING
                                    """, (
                                        profile_id,
                                        param,
                                        row.get('PRES'),
                                        row.get(param),
                                        row.get(f'{param}_ADJUSTED', row.get(param)),
                                        row.get(f'{param}_QC', 1),
                                        row.get(f'{param}_DATA_MODE', 'R')
                                    ))
                
                conn.commit()
                logger.info(f"Stored {len(profiles)} profiles for float {float_id}")
                
            self.db_pool.putconn(conn)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data for float {float_id}: {e}")
            if conn:
                conn.rollback()
                self.db_pool.putconn(conn)
            return False
    
    def process_single_float(self, float_id, include_bgc=False):
        """Process a single float: fetch, process, and store"""
        try:
            # Try physical data first
            ds = self.fetch_float_data(float_id, 'phy')
            if ds is not None:
                df = self.process_float_dataset(ds, float_id, include_bgc=False)
                if not df.empty:
                    self.store_float_data(df, float_id)
            
            # Try BGC data if requested
            if include_bgc:
                ds_bgc = self.fetch_float_data(float_id, 'bgc')
                if ds_bgc is not None:
                    df_bgc = self.process_float_dataset(ds_bgc, float_id, include_bgc=True)
                    if not df_bgc.empty:
                        self.store_float_data(df_bgc, float_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process float {float_id}: {e}")
            return False
    
    def bulk_ingest_indian_ocean(self, max_floats=100, include_bgc=True, parallel_workers=5):
        """
        Bulk ingestion of Indian Ocean ARGO data
        """
        logger.info("Starting bulk ingestion of Indian Ocean ARGO data")
        
        # Get list of floats
        float_ids = self.get_indian_ocean_floats()[:max_floats]
        logger.info(f"Will process {len(float_ids)} floats with {parallel_workers} parallel workers")
        
        # Process floats in parallel
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit tasks
            future_to_float = {
                executor.submit(self.process_single_float, float_id, include_bgc): float_id 
                for float_id in float_ids
            }
            
            # Process completed tasks
            for future in as_completed(future_to_float):
                float_id = future_to_float[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                        logger.info(f"Successfully processed float {float_id} ({successful}/{len(float_ids)})")
                    else:
                        failed += 1
                        logger.warning(f"Failed to process float {float_id} ({failed} failures)")
                except Exception as e:
                    failed += 1
                    logger.error(f"Exception processing float {float_id}: {e}")
        
        logger.info(f"Bulk ingestion completed: {successful} successful, {failed} failed")
        
        # Generate embeddings for all profiles
        logger.info("Generating embeddings for stored profiles...")
        self.generate_all_embeddings()
        
        return successful, failed
    
    def generate_all_embeddings(self):
        """Generate embeddings for all profiles in database"""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get all profiles with measurements
                cursor.execute("""
                    SELECT p.profile_id, p.platform_number, p.cycle_number, 
                           p.latitude, p.longitude, p.datetime,
                           array_agg(DISTINCT m.parameter_name) as parameters,
                           min(m.pressure) as min_depth,
                           max(m.pressure) as max_depth,
                           count(m.measurement_id) as measurement_count
                    FROM profiles p
                    LEFT JOIN measurements m ON p.profile_id = m.profile_id
                    GROUP BY p.profile_id, p.platform_number, p.cycle_number, 
                             p.latitude, p.longitude, p.datetime
                    HAVING count(m.measurement_id) > 0
                """)
                
                profiles = cursor.fetchall()
                
                if profiles:
                    documents = []
                    metadatas = []
                    ids = []
                    
                    for profile in profiles:
                        param_list = ', '.join(profile['parameters']) if profile['parameters'] else 'unknown'
                        date_str = profile['datetime'].strftime('%Y-%m-%d') if profile['datetime'] else 'unknown'
                        
                        doc_text = f"""
                        Float {profile['platform_number']} profile {profile['cycle_number']} 
                        measured on {date_str} at {profile['latitude']:.2f}°N, {profile['longitude']:.2f}°E.
                        Contains {param_list} data from {profile['min_depth']:.1f} to {profile['max_depth']:.1f} dbar
                        with {profile['measurement_count']} measurements.
                        Available parameters: {param_list}.
                        """.strip()
                        
                        documents.append(doc_text)
                        metadatas.append({
                            'profile_id': profile['profile_id'],
                            'platform_number': profile['platform_number'],
                            'cycle_number': profile['cycle_number'],
                            'latitude': float(profile['latitude']),
                            'longitude': float(profile['longitude']),
                            'datetime': date_str,
                            'parameters': param_list,
                            'min_depth': float(profile['min_depth']),
                            'max_depth': float(profile['max_depth']),
                            'measurement_count': profile['measurement_count']
                        })
                        ids.append(f"profile_{profile['profile_id']}")
                    
                    # Generate embeddings in batches
                    batch_size = 100
                    for i in range(0, len(documents), batch_size):
                        batch_docs = documents[i:i+batch_size]
                        batch_metas = metadatas[i:i+batch_size]
                        batch_ids = ids[i:i+batch_size]
                        
                        embeddings = self.embedding_model.encode(batch_docs)
                        
                        # Store in ChromaDB
                        self.vector_collection.add(
                            embeddings=embeddings.tolist(),
                            documents=batch_docs,
                            metadatas=batch_metas,
                            ids=batch_ids
                        )
                        
                        logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                    
                    logger.info(f"Successfully generated embeddings for {len(documents)} profiles")
            
            self.db_pool.putconn(conn)
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return False

def main():
    """Main function for bulk data ingestion"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'argo_ocean_db',
        'user': 'floatchat_user', 
        'password': '140612',  # Update this
        'port': 5432
    }
    
    # Initialize ingestion pipeline
    logger.info("Initializing ARGO data ingestion pipeline...")
    pipeline = ArgoDataIngestionPipeline(db_config)
    
    # Run bulk ingestion
    logger.info("Starting bulk ingestion for Indian Ocean...")
    successful, failed = pipeline.bulk_ingest_indian_ocean(
        max_floats=50,  # Start with 50 floats for testing
        include_bgc=True,
        parallel_workers=3  # Adjust based on your system
    )
    
    logger.info(f"Ingestion completed: {successful} successful, {failed} failed")
    
    # Print summary statistics
    try:
        conn = pipeline.db_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM profiles")
            profile_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM measurements")
            measurement_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT parameter_name) FROM measurements")
            param_count = cursor.fetchone()[0]
            
            logger.info(f"Database now contains:")
            logger.info(f"- {profile_count} profiles")
            logger.info(f"- {measurement_count} measurements") 
            logger.info(f"- {param_count} unique parameters")
        
        pipeline.db_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to get summary statistics: {e}")

if __name__ == "__main__":
    main()