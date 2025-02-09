import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import kstest
import plotly.express as px
import plotly.graph_objects as go
import openpyxl
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, best_path, cust_path, trx_path):
        self.best_path = best_path
        self.cust_path = cust_path
        self.trx_path = trx_path

    def load_data(self):
        best = pd.read_csv(self.best_path)
        cust = pd.read_csv(self.cust_path)
        trx = pd.read_csv(self.trx_path)
        return best, cust, trx

class DataPreprocessor:
    def __init__(self, best, cust, trx):
        self.best = best
        self.cust = cust
        self.trx = trx
        self.df = None
        self.df_copy = None

    def merge_data(self):
        merged_trx = pd.merge(self.trx, self.cust, on='cb_customer_id', how='inner')
        self.df = pd.merge(merged_trx, self.best, on='unique_customer_id', how='left')
        self.df.drop_duplicates(inplace=True)
        self.df_copy = self.df.copy()
        
    def clean_data(self):
        self.df_copy['gender'] = self.df_copy['gender'].replace('UNKNOWN', np.nan)
      #  self.df_copy.drop(columns=['date_of_birth', "gender"], inplace=True, errors='ignore')
        
    def preprocess_dates(self):
        self.df_copy['transaction_date'] = pd.to_datetime(self.df_copy['transaction_date'])

        
    def feature_engineering(self):
        self.df_copy['branch_count'] = self.df_copy.groupby('unique_customer_id')['cb_branch_id'].transform('nunique')
        self.df_copy['discount_sum'] = self.df_copy.groupby('unique_customer_id')['amount_discount'].transform('sum')
        self.df_copy['Monetary_beforeDis'] = self.df_copy.groupby('unique_customer_id')['amount_before_discount'].transform('sum')
    
    def get_data(self):
        return self.df_copy

class RFMCalculator:
    def __init__(self, df):
        self.df = df
    
    def calculate_rfm(self):
        max_date = self.df['transaction_date'].max()
        rfm_data = self.df.groupby('unique_customer_id').agg({'transaction_date': lambda x: x.max()})
        rfm_data['Recency'] = (max_date - rfm_data['transaction_date']).dt.days
        rfm_data.drop(columns='transaction_date', inplace=True)

        freq_data = self.df.groupby('unique_customer_id').agg({'transaction_date': 'count'}).rename(columns={'transaction_date': 'Frequency'})
        mon_data = self.df.groupby('unique_customer_id').agg({'amount_after_discount': 'sum'}).rename(columns={'amount_after_discount': 'Monetary'})
        
        rfm = rfm_data.join(freq_data).join(mon_data)
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        bins = [0, 1, 2, 3, 5, np.inf]
        labels = [1, 2, 3, 4, 5]
        rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=bins, labels=labels)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        rfm['RF_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)
        
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
        
        rfm['RF_segment'] = rfm['RF_Score'].replace(seg_map, regex=True)
        return rfm

class ChurnModel:
    def __init__(self, df):
        self.df = df
        self.model = XGBClassifier(random_state=42)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', self.model)
        ])

    def prepare_data(self):
        self.df['churn_flag'] = np.where(self.df['RF_segment'] == "Churn", 1, 0)
        self.df = self.df[['branch_count', 'discount_sum', 'Recency', 'Frequency', 'unique_customer_id', 'discount_rate', 'discount_per_branch',
                           'Monetary_beforeDis', 'Monetary', 'avg_amount_spent', 'avg_discount', 'churn_flag', 'Monetary_branch', 'branch_discount_interaction',
                           'spend_ratio', 'discount_impact']]
        self.df.drop_duplicates(inplace=True)

        self.X = self.df[['branch_count', 'discount_sum', 'unique_customer_id', 'discount_rate', 'discount_per_branch',
                          'Monetary_beforeDis', 'Monetary', 'avg_amount_spent', 'avg_discount', 'branch_discount_interaction',
                          'spend_ratio', 'discount_impact']]
        self.y = self.df[['churn_flag']]
        
        self.X.drop_duplicates(inplace=True)
        
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y, shuffle=True)
        
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report

class CLVModel:
    def __init__(self, df):
        self.df = df

    def prepare_data(self):
        self.df["CLV"] = self.df['Monetary']
        self.df.drop_duplicates(inplace=True)
        return self.df

