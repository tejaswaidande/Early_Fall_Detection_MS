import pandas as pd
from pymongo import MongoClient
import logging
from typing import Optional
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mongo_connection(uri: str = "mongodb://localhost:27017/") -> MongoClient:
    """Establish connection to MongoDB."""
    try:
        client = MongoClient(uri)
        client.admin.command('ping')  # Test connection
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def process_questionnaire_scores(collection_name: str = "qs") -> Optional[pd.DataFrame]:
    """
    Process questionnaire scores from MongoDB collection.
    Returns a DataFrame with calculated scores (BDI, EDSS, SF-36).
    Handles duplicate entries by keeping the most recent visit (highest VISITNUM).
    """
    try:
        # 1. Connect to MongoDB
        client = get_mongo_connection()
        db = client.thesis_data
        
        # 2. Query data - include all needed fields
        query = {}
        projection = {
            "usubjid": 1,
            "qstestcd": 1,
            "qsstresn": 1,
            "qsscat": 1,
            "visitnum": 1,  # For handling duplicates
            "visit": 1,     # For debugging/verification
            "_id": 0
        }
        
        logger.info(f"Querying collection {collection_name}...")
        cursor = db[collection_name].find(query, projection)
        
        # 3. Convert to DataFrame
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            logger.warning(f"No data found in collection '{collection_name}'")
            return None

        # 4. Clean and prepare data
        # Rename columns to uppercase for consistency
        df = df.rename(columns={
            'usubjid': 'USUBJID',
            'qstestcd': 'QSTESTCD',
            'qsstresn': 'QSSTRESN',
            'qsscat': 'QSSCAT',
            'visitnum': 'VISITNUM',
            'visit': 'VISIT'
        })

        # Log sample data for verification
        logger.info(f"Found {len(df)} records. Sample data:\n{df.head(3)}")
        
        # 5. Handle duplicate entries
        # Sort by USUBJID, QSTESTCD, and VISITNUM (descending to keep most recent)
        df = df.sort_values(['USUBJID', 'QSTESTCD', 'VISITNUM'], 
                          ascending=[True, True, False])
        
        # Keep only the first (most recent) record for each USUBJID/QSTESTCD pair
        df_deduped = df.drop_duplicates(subset=['USUBJID', 'QSTESTCD'], keep='first')
        
        logger.info(f"After deduplication: {len(df_deduped)} unique records remain")
        
        # 6. Pivot data (one row per patient)
        pivoted = df_deduped.pivot(index='USUBJID', columns='QSTESTCD', values='QSSTRESN')
        
        # 7. Calculate composite scores
        
        # BDI Total (sum of BDI items except total score)
        bdi_items = [col for col in pivoted.columns 
                    if col.startswith('BDI01') and col != 'BDI0122']
        if bdi_items:
            pivoted['BDI_TOTAL'] = pivoted[bdi_items].sum(axis=1)
        else:
            pivoted['BDI_TOTAL'] = np.nan
            logger.warning("No BDI items found for total score calculation")
        
        # EDSS Score
        pivoted['EDSS'] = pivoted.get('EDSS0101', np.nan)
        
        # SF-36 Physical Functioning (mean of items)
        pf_items = df_deduped[
            (df_deduped['QSTESTCD'].str.startswith('R3601', na=False)) &
            (df_deduped['QSSCAT'] == 'PHYSICAL FUNCTIONING')
        ]['QSTESTCD'].unique()
        
        if len(pf_items) > 0:
            pivoted['SF36_PF'] = pivoted[pf_items].mean(axis=1)
        else:
            pivoted['SF36_PF'] = np.nan
            logger.warning("No SF-36 Physical Functioning items found")
        
        # 8. Select final features
        feature_columns = ['EDSS', 'BDI_TOTAL', 'SF36_PF']
        
        # Add any KFSS scores if present
        kfss_items = [col for col in pivoted.columns if col.startswith('KFSS')]
        feature_columns.extend(kfss_items)
        
        features = pivoted[feature_columns]
        
        # 9. Add visit information back to the final output
        visit_info = df_deduped[['USUBJID', 'VISIT', 'VISITNUM']].drop_duplicates()
        features = features.merge(visit_info, on='USUBJID', how='left')
        
        logger.info(f"Successfully processed scores for {len(features)} patients")
        return features.reset_index()

    except Exception as e:
        logger.error(f"Error processing scores: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    # Example usage
    logger.info("Starting score processing...")
    scores_df = process_questionnaire_scores(collection_name="qs")
    
    if scores_df is not None:
        print("\nProcessed Scores:")
        print(scores_df.head())
        
        # Save to CSV
        output_path = "../data/processed_scores.csv"
        scores_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed scores to {output_path}")
    else:
        logger.error("No data was processed")