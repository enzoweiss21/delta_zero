#!/usr/bin/env python3
"""Fetch real Aerodrome pool data from Dune Analytics API."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from dune_client.client import DuneClient

API_KEY = "DlM921AUQ5WkGdavgtNByGwDUl024LAG"

# Query IDs - all use 2986326 for now (update if different)
QUERY_ID_VOLUME = 2986326
QUERY_ID_TVL = 2986326
QUERY_ID_FEES = 2986326
QUERY_ID_OVERVIEW = 2986326


class AerodromeDataFetcher:
    """Fetch historical Aerodrome pool data from Dune."""
    
    def __init__(self, api_key: str = API_KEY):
        self.client = DuneClient(api_key)
    
    def fetch_volume_data(
        self,
        pool_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical volume data.
        
        Returns DataFrame with columns: date, volume_usd
        """
        try:
            query_result = self.client.get_latest_result(QUERY_ID_VOLUME)
            
            if query_result.result and query_result.result.rows:
                df = pd.DataFrame(query_result.result.rows)
                
                # Normalize column names
                if 'day' in df.columns:
                    df['date'] = pd.to_datetime(df['day'])
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by pool if specified
                if pool_address and 'pool' in df.columns:
                    df = df[df['pool'].str.lower() == pool_address.lower()]
                
                # Filter by date range
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                # Ensure volume_usd column
                if 'volume_usd' not in df.columns:
                    if 'volume' in df.columns:
                        df['volume_usd'] = df['volume']
                    else:
                        print(f"Warning: No volume column. Available: {df.columns.tolist()}")
                        return pd.DataFrame(columns=['date', 'volume_usd'])
                
                return df[['date', 'volume_usd']].sort_values('date')
            else:
                print("Warning: No volume data returned")
                return pd.DataFrame(columns=['date', 'volume_usd'])
        except Exception as e:
            print(f"Error fetching volume data: {e}")
            return pd.DataFrame(columns=['date', 'volume_usd'])
    
    def fetch_tvl_data(
        self,
        pool_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical TVL data.
        
        Returns DataFrame with columns: date, tvl_usd
        """
        try:
            query_result = self.client.get_latest_result(QUERY_ID_TVL)
            
            if query_result.result and query_result.result.rows:
                df = pd.DataFrame(query_result.result.rows)
                
                # Normalize column names
                if 'day' in df.columns:
                    df['date'] = pd.to_datetime(df['day'])
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by pool if specified
                if pool_address and 'pool' in df.columns:
                    df = df[df['pool'].str.lower() == pool_address.lower()]
                
                # Filter by date range
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                # Ensure tvl_usd column
                if 'tvl_usd' not in df.columns:
                    if 'tvl' in df.columns:
                        df['tvl_usd'] = df['tvl']
                    else:
                        print(f"Warning: No TVL column. Available: {df.columns.tolist()}")
                        return pd.DataFrame(columns=['date', 'tvl_usd'])
                
                return df[['date', 'tvl_usd']].sort_values('date')
            else:
                print("Warning: No TVL data returned")
                return pd.DataFrame(columns=['date', 'tvl_usd'])
        except Exception as e:
            print(f"Error fetching TVL data: {e}")
            return pd.DataFrame(columns=['date', 'tvl_usd'])
    
    def fetch_swap_fees_data(
        self,
        pool_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical swap fees data.
        
        Returns DataFrame with columns: date, fees_usd
        """
        try:
            query_result = self.client.get_latest_result(QUERY_ID_FEES)
            
            if query_result.result and query_result.result.rows:
                df = pd.DataFrame(query_result.result.rows)
                
                # Normalize column names
                if 'day' in df.columns:
                    df['date'] = pd.to_datetime(df['day'])
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by pool if specified
                if pool_address and 'pool' in df.columns:
                    df = df[df['pool'].str.lower() == pool_address.lower()]
                
                # Filter by date range
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                # Ensure fees_usd column
                if 'fees_usd' not in df.columns:
                    if 'fees' in df.columns:
                        df['fees_usd'] = df['fees']
                    elif 'swap_fees' in df.columns:
                        df['fees_usd'] = df['swap_fees']
                    else:
                        print(f"Warning: No fees column. Available: {df.columns.tolist()}")
                        return pd.DataFrame(columns=['date', 'fees_usd'])
                
                return df[['date', 'fees_usd']].sort_values('date')
            else:
                print("Warning: No swap fees data returned")
                return pd.DataFrame(columns=['date', 'fees_usd'])
        except Exception as e:
            print(f"Error fetching swap fees data: {e}")
            return pd.DataFrame(columns=['date', 'fees_usd'])
    
    def fetch_all_pool_data(
        self,
        pool_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all pool data (volume, TVL, fees) and return as dict.
        
        Returns:
            {
                'volume': DataFrame with date, volume_usd,
                'tvl': DataFrame with date, tvl_usd,
                'fees': DataFrame with date, fees_usd
            }
        """
        return {
            'volume': self.fetch_volume_data(pool_address, start_date, end_date),
            'tvl': self.fetch_tvl_data(pool_address, start_date, end_date),
            'fees': self.fetch_swap_fees_data(pool_address, start_date, end_date),
        }


def test_dune_connection():
    """Test connection to Dune API."""
    try:
        fetcher = AerodromeDataFetcher()
        print("Testing Dune API connection...")
        
        # Test TVL
        print("\nFetching TVL data...")
        tvl_df = fetcher.fetch_tvl_data()
        print(f"TVL data shape: {tvl_df.shape}")
        if not tvl_df.empty:
            print(f"Columns: {tvl_df.columns.tolist()}")
            print(f"Date range: {tvl_df['date'].min()} to {tvl_df['date'].max()}")
            print(f"\nFirst 5 rows:\n{tvl_df.head()}")
        else:
            print("No TVL data returned")
        
        # Test Volume
        print("\nFetching Volume data...")
        volume_df = fetcher.fetch_volume_data()
        print(f"Volume data shape: {volume_df.shape}")
        if not volume_df.empty:
            print(f"Columns: {volume_df.columns.tolist()}")
            print(f"Date range: {volume_df['date'].min()} to {volume_df['date'].max()}")
            print(f"\nFirst 5 rows:\n{volume_df.head()}")
        else:
            print("No volume data returned")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dune_connection()
