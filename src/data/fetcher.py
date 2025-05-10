"""
Module pour récupérer les données des actions depuis différentes sources
"""

import os
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta


class DataFetcher:
    """
    Classe pour récupérer les données des actions depuis différentes sources
    """
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialise le DataFetcher
        
        Args:
            cache_dir: Répertoire où stocker les données récupérées
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_sp500_symbols(self) -> List[str]:
        """
        Récupère la liste des symboles du S&P 500
        
        Returns:
            Liste des symboles du S&P 500
        """
        # Utilisation de Wikipedia pour récupérer la liste des symboles du S&P 500
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    
    def get_historical_data(self, 
                          symbols: Union[str, List[str]], 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          period: str = "1y",
                          interval: str = "1d",
                          use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Récupère les données historiques pour les symboles spécifiés
        
        Args:
            symbols: Liste des symboles ou symbole unique
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD'
            period: Période (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Intervalle (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Utiliser les données en cache si disponibles
            
        Returns:
            Dictionnaire de DataFrames avec les données historiques pour chaque symbole
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Date de fin par défaut = aujourd'hui
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Si start_date n'est pas fourni mais period l'est, on utilisera period
        use_period = start_date is None
        
        results = {}
        
        for symbol in symbols:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{period}_{interval}.csv")
            
            # Vérifier si les données en cache sont disponibles et récentes
            if use_cache and os.path.exists(cache_file):
                cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Vérifier si les données en cache sont à jour
                if not cached_data.empty and cached_data.index[-1].strftime("%Y-%m-%d") >= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                    results[symbol] = cached_data
                    continue
            
            try:
                # Récupérer les données depuis Yahoo Finance
                if use_period:
                    data = yf.download(symbol, period=period, interval=interval, progress=False)
                else:
                    data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
                
                if not data.empty:
                    # Sauvegarde en cache
                    data.to_csv(cache_file)
                    results[symbol] = data
                else:
                    print(f"Aucune donnée disponible pour {symbol}")
            except Exception as e:
                print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        
        return results
    
    def get_fundamental_data(self, symbols: Union[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Récupère les données fondamentales pour les symboles spécifiés
        
        Args:
            symbols: Liste des symboles ou symbole unique
            
        Returns:
            Dictionnaire de DataFrames avec les données fondamentales pour chaque symbole
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        results = {}
        
        for symbol in symbols:
            try:
                # Créer un objet Ticker
                ticker = yf.Ticker(symbol)
                
                # Récupérer les informations fondamentales
                info = ticker.info
                
                # Convertir en DataFrame
                df = pd.DataFrame([info])
                
                results[symbol] = df
            except Exception as e:
                print(f"Erreur lors de la récupération des données fondamentales pour {symbol}: {e}")
        
        return results
