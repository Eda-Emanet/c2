import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

class DataProcessor:
    def __init__(self, best_path: str, cust_path: str, trx_path: str):
        """
        Initialize the DataProcessor with paths to required CSV files.
        
        Args:
            best_path (str): Path to the best customer sample CSV
            cust_path (str): Path to the customer sample CSV 
            trx_path (str): Path to the transaction sample CSV
        """
        self.best = pd.read_csv(best_path)
        self.cust = pd.read_csv(cust_path)
        self.trx = pd.read_csv(trx_path)
        self.df = None
        self.df_copy = None
        self.rfm = None

    def load_and_merge_data(self) -> None:
        """Merge the transaction, customer and best customer data."""
        merged_trx = pd.merge(self.trx, self.cust, on='cb_customer_id', how='inner')
        self.df = pd.merge(merged_trx, self.best, on='unique_customer_id', how='left')
        self.df.drop_duplicates(inplace=True)
        self.df_copy = self.df.copy()

    def clean_gender_data(self) -> None:
        """Replace 'UNKNOWN' gender values with NaN."""
        self.df_copy['gender'] = self.df_copy['gender'].replace('UNKNOWN', np.nan)

    def convert_date_columns(self) -> None:
        """Convert date columns to datetime format."""
        self.df_copy['transaction_date'] = pd.to_datetime(self.df_copy['transaction_date'])
        self.df_copy['date_of_birth'] = pd.to_datetime(self.df_copy['date_of_birth'])

    def drop_unnecessary_columns(self) -> None:
        """Remove date_of_birth and gender columns."""
        self.df_copy.drop(columns=['date_of_birth', 'gender'], inplace=True)

    def add_feature_columns(self) -> None:
        """Add new feature columns based on customer behavior."""
        # Branch count per customer
        self.df_copy['branch_count'] = self.df_copy.groupby('unique_customer_id')['cb_branch_id'].transform('nunique')
        
        # Total discount per customer
        self.df_copy['discount_sum'] = self.df_copy.groupby('unique_customer_id')['amount_discount'].transform('sum')
        
        # Total pre-discount spending per customer
        self.df_copy['Monetary_beforeDis'] = self.df_copy.groupby('unique_customer_id')['amount_before_discount'].transform('sum')

    def calculate_rfm(self) -> None:
        """Calculate RFM (Recency, Frequency, Monetary) metrics."""
        max_date = self.df_copy['transaction_date'].max()

        # Calculate Recency
        rfm_data = self.df_copy.groupby('unique_customer_id').agg({
            'transaction_date': lambda x: x.max(),
        })
        rfm_data['Recency'] = (max_date - rfm_data['transaction_date']).dt.days
        rfm_data.drop(columns='transaction_date', inplace=True)

        # Calculate Frequency
        freq_data = self.df_copy.groupby('unique_customer_id').agg({
            'transaction_date': 'count'
        }).rename(columns={'transaction_date': 'Frequency'})

        # Calculate Monetary
        mon_data = self.df_copy.groupby('unique_customer_id').agg({
            'amount_after_discount': 'sum'
        }).rename(columns={'amount_after_discount': 'Monetary'})

        # Combine RFM metrics
        self.rfm = rfm_data.join(freq_data).join(mon_data)

    def assign_rfm_scores(self) -> None:
        """Assign RFM scores and segments."""
        if self.rfm is None:
            raise ValueError("RFM metrics must be calculated first")

        # Recency scoring
        self.rfm['R_Score'] = pd.qcut(self.rfm['Recency'], 5, labels=[5,4,3,2,1])
        
        # Frequency scoring
        bins = [0, 1, 2, 3, 5, np.inf]
        labels = [1, 2, 3, 4, 5]
        self.rfm['F_Score'] = pd.cut(self.rfm['Frequency'], bins=bins, labels=labels)
        
        # Monetary scoring
        self.rfm['M_Score'] = pd.qcut(self.rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Combined RF score
        self.rfm['RF_Score'] = self.rfm['R_Score'].astype(str) + self.rfm['F_Score'].astype(str)

        # Segment mapping
        seg_map = {
            r'[1-2][1-2]': 'Churn',
            r'[1-2][3-4]': 'at_Risk',
            r'[1-2]5': 'cant_loose',
            r'3[1-2]': 'about_to_sleep',
            r'33': 'need_attention',
            r'[3-4][4-5]': 'loyal_customers',
            r'41': 'promising',
            r'51': 'new_customers',
            r'[4-5][2-3]': 'potential_loyalists',
            r'5[4-5]': 'champions'
        }
        
        self.rfm['RF_segment'] = self.rfm['RF_Score'].replace(seg_map, regex=True)



    def process_data(self) -> None:
        """Execute the complete data processing pipeline."""
        self.load_and_merge_data()
        self.clean_gender_data()
        self.convert_date_columns()
        self.drop_unnecessary_columns()
        self.add_feature_columns()
        self.calculate_rfm()
        self.assign_rfm_scores()

    def get_rfm_data(self) -> pd.DataFrame:
        """Return the processed RFM data."""
        if self.rfm is None:
            raise ValueError("Data has not been processed yet. Run process_data() first.")
        return self.rfm