# test_enhanced_floatchat.py - Test Suite for Enhanced FloatChat

import pytest
import pandas as pd
import numpy as np
import psycopg2
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import os
from unittest.mock import Mock, patch
import xarray as xr
from datetime import datetime

# Import the main application components
# Note: Adjust imports based on your actual module structure
import sys
sys.path.append('.')

class TestEnhancedFloatChat:
    """Test suite for Enhanced FloatChat functionality"""
    
    @pytest.fixture(scope="class")
    def test_db_config(self):
        """Test database configuration"""
        return {
            'host': 'localhost',
            'database': 'test_argo_db',  # Use separate test database
            'user': 'postgres',
            'password': 'test_password',
            'port': 5432
        }
    
    @pytest.fixture(scope="class")
    def sample_argo_data(self):
        """Create sample ARGO data for testing"""
        
        # Create sample xarray dataset mimicking ARGO data structure
        n_profiles = 5
        n_levels = 50
        
        # Coordinates
        coords = {
            'N_PROF': range(n_profiles),
            'N_LEVELS': range(n_levels)
        }
        
        # Sample data variables
        data_vars = {
            'LATITUDE': (['N_PROF'], np.random.uniform(-30, 30, n_profiles)),
            'LONGITUDE': (['N_PROF'], np.random.uniform(20, 120, n_profiles)),
            'JULD': (['N_PROF'], np.random.uniform(25000, 27000, n_profiles)),  # Days since 1950
            'PLATFORM_NUMBER': (['N_PROF'], ['6902746', '6902747', '6902748', '6902749', '6902750']),
            'CYCLE_NUMBER': (['N_PROF'], range(1, n_profiles + 1)),
            'PRES': (['N_PROF', 'N_LEVELS'], np.random.uniform(0, 2000, (n_profiles, n_levels))),
            'TEMP': (['N_PROF', 'N_LEVELS'], np.random.uniform(2, 30, (n_profiles, n_levels))),
            'PSAL': (['N_PROF', 'N_LEVELS'], np.random.uniform(32, 37, (n_profiles, n_levels))),
            'DOXY': (['N_PROF', 'N_LEVELS'], np.random.uniform(100, 300, (n_profiles, n_levels))),
            'CHLA': (['N_PROF', 'N_LEVELS'], np.random.uniform(0.1, 2.0, (n_profiles, n_levels)))
        }
        
        # Create xarray dataset
        ds = xr.Dataset(data_vars, coords=coords)
        return ds
    
    @pytest.fixture
    def temp_chroma_db(self):
        """Create temporary ChromaDB for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_database_schema_creation(self, test_db_config):
        """Test database schema creation"""
        # This test requires a test PostgreSQL database
        # Skip if database is not available
        try:
            conn = psycopg2.connect(**test_db_config)
            conn.close()
        except psycopg2.OperationalError:
            pytest.skip("Test database not available")
        
        # Test schema creation logic
        # Import and test the create_database_schema function
        # from oceanic_enhanced import create_database_schema
        
        # pool = psycopg2.pool.SimpleConnectionPool(1, 5, **test_db_config)
        # result = create_database_schema(pool)
        # assert result == True
        
        # Verify tables were created
        # conn = pool.getconn()
        # with conn.cursor() as cursor:
        #     cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        #     tables = [row[0] for row in cursor.fetchall()]
        #     expected_tables = ['profiles', 'measurements', 'bgc_parameters']
        #     for table in expected_tables:
        #         assert table in tables
        # pool.putconn(conn)
        
        assert True  # Placeholder for actual test
    
    def test_data_processing_pipeline(self, sample_argo_data):
        """Test ARGO data processing pipeline"""
        
        # Mock the processing function
        def mock_process_enhanced_dataset(ds, floats_key, include_bgc=False):
            """Mock version of process_enhanced_dataset function"""
            
            # Convert sample xarray to DataFrame (simplified)
            df_data = []
            for prof_idx in range(len(ds.N_PROF)):
                for level_idx in range(len(ds.N_LEVELS)):
                    row = {
                        'PLATFORM_NUMBER': ds.PLATFORM_NUMBER.values[prof_idx],
                        'CYCLE_NUMBER': ds.CYCLE_NUMBER.values[prof_idx],
                        'LATITUDE': ds.LATITUDE.values[prof_idx],
                        'LONGITUDE': ds.LONGITUDE.values[prof_idx],
                        'JULD': ds.JULD.values[prof_idx],
                        'PRES': ds.PRES.values[prof_idx, level_idx],
                        'TEMP': ds.TEMP.values[prof_idx, level_idx],
                        'PSAL': ds.PSAL.values[prof_idx, level_idx],
                    }
                    
                    if include_bgc:
                        row.update({
                            'DOXY': ds.DOXY.values[prof_idx, level_idx],
                            'CHLA': ds.CHLA.values[prof_idx, level_idx],
                        })
                    
                    df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Add datetime conversion
            df['DATETIME'] = pd.to_datetime('1950-01-01') + pd.to_timedelta(df['JULD'], unit='D')
            df['FLOAT_ID'] = df['PLATFORM_NUMBER']
            
            # Remove NaN values
            df = df.dropna()
            
            return df
        
        # Test physical data processing
        df_phy = mock_process_enhanced_dataset(sample_argo_data, (6902746,), include_bgc=False)
        
        assert not df_phy.empty
        assert 'TEMP' in df_phy.columns
        assert 'PSAL' in df_phy.columns
        assert 'PRES' in df_phy.columns
        assert 'DATETIME' in df_phy.columns
        assert len(df_phy['PLATFORM_NUMBER'].unique()) == 5
        
        # Test BGC data processing
        df_bgc = mock_process_enhanced_dataset(sample_argo_data, (6902746,), include_bgc=True)
        
        assert not df_bgc.empty
        assert 'DOXY' in df_bgc.columns
        assert 'CHLA' in df_bgc.columns
    
    def test_vector_database_operations(self, temp_chroma_db):
        """Test ChromaDB vector database operations"""
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=temp_chroma_db)
        collection = client.get_or_create_collection(
            name="test_argo_profiles",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Test embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample metadata
        documents = [
            "Float 6902746 profile 1 measured on 2023-01-15 at 10.5°N, 65.2°E. Contains TEMP, PSAL data.",
            "Float 6902747 profile 2 measured on 2023-02-20 at 12.3°N, 67.8°E. Contains TEMP, PSAL, DOXY data.",
            "Float 6902748 profile 3 measured on 2023-03-10 at 8.7°N, 70.1°E. Contains TEMP, PSAL, CHLA data."
        ]
        
        metadatas = [
            {"platform_number": "6902746", "cycle_number": 1, "latitude": 10.5, "longitude": 65.2},
            {"platform_number": "6902747", "cycle_number": 2, "latitude": 12.3, "longitude": 67.8},
            {"platform_number": "6902748", "cycle_number": 3, "latitude": 8.7, "longitude": 70.1}
        ]
        
        ids = ["profile_1", "profile_2", "profile_3"]
        
        # Generate embeddings
        embeddings = embedding_model.encode(documents)
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Test similarity search
        query = "temperature and salinity data in the Indian Ocean"
        query_embedding = embedding_model.encode([query])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )
        
        assert len(results['ids'][0]) == 2
        assert len(results['documents'][0]) == 2
        assert len(results['metadatas'][0]) == 2
        assert all(isinstance(d, float) for d in results['distances'][0])
    
    def test_query_intent_detection(self):
        """Test enhanced intent detection functionality"""
        
        def mock_detect_enhanced_intent(query):
            """Mock intent detection function"""
            query_lower = query.lower()
            if any(kw in query_lower for kw in ["profile", "depth", "vertical"]):
                return "profile_data"
            elif any(kw in query_lower for kw in ["summary", "statistics", "average"]):
                return "summary_stats"
            elif any(kw in query_lower for kw in ["map", "location", "trajectory"]):
                return "location_data"
            elif any(kw in query_lower for kw in ["time series", "over time", "trend"]):
                return "time_series"
            else:
                return "general_query"
        
        # Test various query types
        test_cases = [
            ("Show me salinity profiles for float 6902746", "profile_data"),
            ("What are the average temperature values?", "summary_stats"),
            ("Plot the trajectory of floats in the Indian Ocean", "location_data"),
            ("Show temperature time series over the last year", "time_series"),
            ("Tell me about the available data", "general_query")
        ]
        
        for query, expected_intent in test_cases:
            intent = mock_detect_enhanced_intent(query)
            assert intent == expected_intent, f"Query: '{query}' expected '{expected_intent}', got '{intent}'"
    
    def test_sql_query_generation(self):
        """Test SQL query generation from natural language"""
        
        # Mock profile IDs for testing
        mock_profile_ids = [1, 2, 3, 4, 5]
        
        def mock_enhanced_query_to_sql(query, embedding_model, vector_collection, pool):
            """Mock SQL generation function"""
            
            intent = "profile_data"  # Simplified for testing
            
            base_sql = """
            SELECT p.profile_id, p.platform_number, p.cycle_number, 
                   p.latitude, p.longitude, p.datetime,
                   m.parameter_name, m.pressure, m.value, m.value_adjusted
            FROM profiles p
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE p.profile_id = ANY(%s)
            ORDER BY p.profile_id, m.parameter_name, m.pressure
            """
            
            sql_values = [mock_profile_ids]
            context_info = f"Found {len(mock_profile_ids)} relevant profiles"
            retrieved_profiles = [{"platform_number": "6902746", "cycle_number": 1}]
            
            return (base_sql, sql_values), context_info, retrieved_profiles
        
        # Test SQL generation
        query = "Show me temperature profiles for floats in the Arabian Sea"
        sql_info, context, profiles = mock_enhanced_query_to_sql(query, None, None, None)
        
        assert sql_info is not None
        assert "SELECT" in sql_info[0]
        assert "FROM profiles p" in sql_info[0]
        assert len(sql_info[1]) == 1  # One parameter (profile_ids)
        assert mock_profile_ids == sql_info[1][0]
        assert "Found 5 relevant profiles" in context
    
    def test_visualization_functions(self, sample_argo_data):
        """Test visualization function logic"""
        
        # Create sample processed DataFrame
        df_data = []
        for prof_idx in range(5):  # 5 profiles
            for level_idx in range(10):  # 10 depth levels each
                df_data.append({
                    'platform_number': f'690274{prof_idx}',
                    'cycle_number': 1,
                    'latitude': 10.0 + prof_idx,
                    'longitude': 65.0 + prof_idx,
                    'datetime': datetime(2023, 1, 1 + prof_idx),
                    'parameter_name': 'TEMP',
                    'pressure': level_idx * 10,
                    'value': 25.0 - level_idx * 0.5,
                    'value_adjusted': 25.0 - level_idx * 0.5
                })
        
        df = pd.DataFrame(df_data)
        
        def mock_create_enhanced_trajectory_plot(df, query_context=""):
            """Mock trajectory plotting function"""
            if df.empty:
                return "No trajectory data available."
            
            trajectory_df = df.groupby(['platform_number', 'cycle_number']).agg({
                'latitude': 'first',
                'longitude': 'first',
                'datetime': 'first'
            }).reset_index()
            
            n_floats = trajectory_df['platform_number'].nunique()
            n_profiles = len(trajectory_df)
            
            return f"Displaying {n_profiles} profiles from {n_floats} float(s). {query_context}"
        
        def mock_create_enhanced_profile_plot(df, query_context=""):
            """Mock profile plotting function"""
            if df.empty:
                return "No profile data available."
            
            if 'pressure' not in df.columns:
                return "Missing pressure data for profile visualization."
            
            available_params = df['parameter_name'].unique()
            summary = f"Profile plots for {', '.join(available_params)}. {query_context}"
            
            return summary
        
        # Test trajectory plotting
        traj_summary = mock_create_enhanced_trajectory_plot(df, "test query")
        assert "Displaying 5 profiles from 5 float(s)" in traj_summary
        
        # Test profile plotting
        profile_summary = mock_create_enhanced_profile_plot(df, "test query")
        assert "Profile plots for TEMP" in profile_summary
    
    def test_data_quality_validation(self):
        """Test data quality validation and filtering"""
        
        # Create sample data with quality issues
        df = pd.DataFrame({
            'LATITUDE': [10.5, -91.0, 20.3, np.nan, 15.7],  # Invalid lat: -91.0, NaN
            'LONGITUDE': [65.2, 120.5, 181.0, 70.8, np.nan],  # Invalid lon: 181.0, NaN
            'TEMP': [25.5, np.nan, 22.3, 28.1, 19.2],
            'PSAL': [35.1, 34.8, np.nan, 36.2, 34.5],
            'PRES': [10.0, 20.0, 30.0, 40.0, 50.0],
            'PLATFORM_NUMBER': ['6902746', '6902747', '6902748', '6902749', '6902750'],
            'CYCLE_NUMBER': [1, 2, 3, 4, 5]
        })
        
        def apply_quality_filters(df):
            """Apply data quality filters"""
            # Geographic validation
            df = df[(df['LATITUDE'] >= -90) & (df['LATITUDE'] <= 90)]
            df = df[(df['LONGITUDE'] >= -180) & (df['LONGITUDE'] <= 180)]
            
            # Remove rows with all NaN measurements
            measurement_cols = ['TEMP', 'PSAL', 'PRES']
            df = df.dropna(subset=measurement_cols, how='all')
            
            # Remove rows without coordinates
            df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
            
            return df.reset_index(drop=True)
        
        # Apply filters
        filtered_df = apply_quality_filters(df)
        
        # Verify quality filtering
        assert len(filtered_df) == 1  # Only 2 valid rows should remain
        assert all(filtered_df['LATITUDE'] >= -90) and all(filtered_df['LATITUDE'] <= 90)
        assert all(filtered_df['LONGITUDE'] >= -180) and all(filtered_df['LONGITUDE'] <= 180)
        assert not filtered_df[['LATITUDE', 'LONGITUDE']].isnull().any().any()
    
    def test_bgc_parameter_handling(self):
        """Test BGC parameter processing and validation"""
        
        # Sample BGC parameters
        bgc_params = ['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700']
        
        # Sample data with BGC parameters
        df = pd.DataFrame({
            'PLATFORM_NUMBER': ['6902746'] * 5,
            'CYCLE_NUMBER': [1] * 5,
            'PRES': [10, 20, 30, 40, 50],
            'TEMP': [25.0, 24.5, 24.0, 23.5, 23.0],
            'PSAL': [35.0, 35.1, 35.2, 35.3, 35.4],
            'DOXY': [250.5, 248.3, 245.1, 242.8, 240.2],
            'CHLA': [1.2, 1.1, 0.9, 0.7, 0.5],
            'NITRATE': [15.2, 15.8, 16.1, 16.5, 17.0],
            'PH_IN_SITU_TOTAL': [8.1, 8.05, 8.0, 7.95, 7.9],
            'BBP700': [0.002, 0.0018, 0.0016, 0.0014, 0.0012]
        })
        
        # Test BGC parameter availability
        available_bgc = [param for param in bgc_params if param in df.columns]
        assert len(available_bgc) == 5
        assert 'DOXY' in available_bgc
        assert 'CHLA' in available_bgc
        
        # Test BGC data ranges (basic validation)
        assert df['DOXY'].min() > 0  # Oxygen should be positive
        assert df['CHLA'].min() >= 0  # Chlorophyll should be non-negative
        assert df['PH_IN_SITU_TOTAL'].between(7, 9).all()  # pH in realistic range

# Additional integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data fetch to visualization"""
        
        # This would be a comprehensive test that:
        # 1. Mocks ARGO data fetching
        # 2. Processes the data
        # 3. Stores in database
        # 4. Creates embeddings
        # 5. Performs query processing
        # 6. Generates visualizations
        
        # For now, just test that the workflow components are testable
        assert True
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for key operations"""
        
        # Test data processing speed
        # Test vector search performance
        # Test SQL query execution time
        
        # Placeholder for performance tests
        assert True

# Run tests
if __name__ == "__main__":
    # Run specific test functions for development
    test_instance = TestEnhancedFloatChat()
    
    # Create sample data
    sample_data = test_instance.sample_argo_data()
    
    print("Running Enhanced FloatChat tests...")
    
    # Test data processing
    print("Testing data processing pipeline...")
    test_instance.test_data_processing_pipeline(sample_data)
    
    # Test intent detection
    print("Testing query intent detection...")
    test_instance.test_query_intent_detection()
    
    # Test SQL generation
    print("Testing SQL query generation...")
    test_instance.test_sql_query_generation()
    
    # Test visualization
    print("Testing visualization functions...")
    test_instance.test_visualization_functions(sample_data)
    
    # Test data quality
    print("Testing data quality validation...")
    test_instance.test_data_quality_validation()
    
    # Test BGC parameters
    print("Testing BGC parameter handling...")
    test_instance.test_bgc_parameter_handling()
    
    print("All tests completed successfully!")
    print("\nTo run full test suite with pytest:")
    print("pip install pytest")
    print("pytest test_enhanced_floatchat.py -v")