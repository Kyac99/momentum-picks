"""
Module contenant des fonctions utilitaires pour le projet
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import xlsxwriter


def calculate_volatility(data: pd.DataFrame, window: int = 20) -> float:
    """
    Calcule la volatilité (ATR - Average True Range) d'une action
    
    Args:
        data: DataFrame avec les colonnes OHLC
        window: Fenêtre pour le calcul de l'ATR
        
    Returns:
        Valeur de l'ATR
    """
    if len(data) < window:
        return np.nan
    
    # S'assurer que toutes les colonnes nécessaires sont présentes
    required_columns = ['High', 'Low', 'Close']
    if not all(col in data.columns for col in required_columns):
        return np.nan
    
    # Calculer les True Ranges
    data = data.copy()
    data['prev_close'] = data['Close'].shift(1)
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = abs(data['High'] - data['prev_close'])
    data['tr3'] = abs(data['Low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculer l'ATR
    atr = data['true_range'].rolling(window=window).mean().iloc[-1]
    
    return atr


def calculate_volatility_dict(data_dict: Dict[str, pd.DataFrame], window: int = 20) -> Dict[str, float]:
    """
    Calcule la volatilité (ATR) pour un dictionnaire de DataFrames
    
    Args:
        data_dict: Dictionnaire de DataFrames avec les colonnes OHLC
        window: Fenêtre pour le calcul de l'ATR
        
    Returns:
        Dictionnaire avec l'ATR pour chaque symbole
    """
    volatility_dict = {}
    
    for symbol, data in data_dict.items():
        volatility_dict[symbol] = calculate_volatility(data, window)
    
    return volatility_dict


def export_to_excel(data: pd.DataFrame, 
                   file_path: str, 
                   sheet_name: str = "Results", 
                   format_dict: Optional[Dict[str, str]] = None):
    """
    Exporte un DataFrame vers un fichier Excel avec formatage
    
    Args:
        data: DataFrame à exporter
        file_path: Chemin du fichier Excel
        sheet_name: Nom de la feuille
        format_dict: Dictionnaire spécifiant les formats pour certaines colonnes
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Créer un writer Excel
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    
    # Exporter le DataFrame
    data.to_excel(writer, sheet_name=sheet_name, index=True)
    
    # Formatage
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # Formats par défaut
    formats = {
        'percentage': workbook.add_format({'num_format': '0.00%'}),
        'dollar': workbook.add_format({'num_format': '$#,##0.00'}),
        'number': workbook.add_format({'num_format': '0.00'}),
        'integer': workbook.add_format({'num_format': '0'}),
        'date': workbook.add_format({'num_format': 'yyyy-mm-dd'}),
        'header': workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'}),
    }
    
    # Appliquer le format d'en-tête
    for col_num, value in enumerate(data.columns.values):
        worksheet.write(0, col_num + 1, value, formats['header'])
    
    # Appliquer les formats spécifiques aux colonnes
    if format_dict:
        for col, fmt in format_dict.items():
            if col in data.columns and fmt in formats:
                col_idx = data.columns.get_loc(col)
                worksheet.set_column(col_idx + 1, col_idx + 1, None, formats[fmt])
    
    # Ajuster la largeur des colonnes
    for i, col in enumerate(data.columns):
        max_len = max(
            data[col].astype(str).str.len().max(),  # Longueur max des données
            len(str(col))  # Longueur de l'en-tête
        )
        worksheet.set_column(i + 1, i + 1, max_len + 2)  # +2 pour un peu d'espace supplémentaire
    
    # Sauvegarder le fichier Excel
    writer.close()
    
    print(f"Exporté avec succès vers {file_path}")


def filter_stocks_by_market_cap(symbols: List[str], 
                              min_market_cap: Optional[float] = None,
                              max_market_cap: Optional[float] = None,
                              data_fetcher=None) -> List[str]:
    """
    Filtre une liste de symboles par capitalisation boursière
    
    Args:
        symbols: Liste des symboles à filtrer
        min_market_cap: Capitalisation boursière minimale (en dollars)
        max_market_cap: Capitalisation boursière maximale (en dollars)
        data_fetcher: Instance de DataFetcher pour récupérer les données
        
    Returns:
        Liste filtrée de symboles
    """
    if data_fetcher is None:
        from src.data.fetcher import DataFetcher
        data_fetcher = DataFetcher()
    
    if min_market_cap is None and max_market_cap is None:
        return symbols
    
    # Récupérer les données fondamentales
    fundamental_data = data_fetcher.get_fundamental_data(symbols)
    
    filtered_symbols = []
    
    for symbol, data in fundamental_data.items():
        if 'marketCap' in data.columns:
            market_cap = data['marketCap'].iloc[0]
            
            # Appliquer les filtres
            if min_market_cap is not None and market_cap < min_market_cap:
                continue
                
            if max_market_cap is not None and market_cap > max_market_cap:
                continue
                
            filtered_symbols.append(symbol)
    
    return filtered_symbols


def calculate_correlation_matrix(price_data: Dict[str, pd.DataFrame], 
                               symbols: List[str],
                               window: int = 90) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation entre les rendements des actions
    
    Args:
        price_data: Dictionnaire de DataFrames avec les données de prix
        symbols: Liste des symboles à inclure
        window: Fenêtre temporelle pour calculer les corrélations
        
    Returns:
        DataFrame avec la matrice de corrélation
    """
    # Extraire les rendements de tous les symboles
    returns_dict = {}
    
    for symbol in symbols:
        if symbol in price_data and 'Close' in price_data[symbol].columns:
            # Calculer les rendements quotidiens
            df = price_data[symbol].copy()
            df = df.iloc[-window:] if len(df) > window else df
            returns = df['Close'].pct_change().dropna()
            returns_dict[symbol] = returns
    
    # Créer un DataFrame avec tous les rendements
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculer la matrice de corrélation
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def backtest_momentum_strategy(price_data: Dict[str, pd.DataFrame],
                             momentum_calculator,
                             start_date: str,
                             end_date: str,
                             rebalance_period: int = 30,  # En jours
                             n_stocks: int = 50,
                             initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Effectue un backtest simple de la stratégie de momentum
    
    Args:
        price_data: Dictionnaire de DataFrames avec les données de prix
        momentum_calculator: Instance de MomentumCalculator
        start_date: Date de début au format 'YYYY-MM-DD'
        end_date: Date de fin au format 'YYYY-MM-DD'
        rebalance_period: Période de rééquilibrage en jours
        n_stocks: Nombre d'actions à inclure dans le portefeuille
        initial_capital: Capital initial
        
    Returns:
        DataFrame avec les résultats du backtest
    """
    # Convertir les dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Créer une liste de dates de rééquilibrage
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=f'{rebalance_period}D')
    
    # DataFrame pour stocker les résultats
    results = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    results['portfolio_value'] = np.nan
    results.loc[start_date, 'portfolio_value'] = initial_capital
    
    # Pour chaque date de rééquilibrage
    current_portfolio = {}
    current_capital = initial_capital
    last_rebalance_date = None
    
    for rebalance_date in rebalance_dates:
        print(f"Rééquilibrage à la date: {rebalance_date}")
        
        # Filtrer les données jusqu'à la date de rééquilibrage
        filtered_data = {}
        for symbol, df in price_data.items():
            mask = (df.index <= rebalance_date)
            filtered_data[symbol] = df[mask].copy()
        
        # Calculer les scores de momentum
        try:
            momentum_df = momentum_calculator.calculate_momentum_percentile(filtered_data)
            top_stocks = momentum_calculator.get_top_momentum_stocks(momentum_df, n=n_stocks)
            
            # Si c'est le premier rééquilibrage, initialiser directement
            if last_rebalance_date is None:
                position_size = current_capital / len(top_stocks)
                
                for symbol in top_stocks:
                    if symbol in filtered_data and not filtered_data[symbol].empty:
                        price = filtered_data[symbol]['Close'].iloc[-1]
                        shares = position_size / price
                        current_portfolio[symbol] = {'shares': shares, 'entry_price': price}
            else:
                # Calculer la valeur actuelle du portefeuille
                current_value = 0
                for symbol, position in current_portfolio.items():
                    if symbol in filtered_data and not filtered_data[symbol].empty:
                        price = filtered_data[symbol]['Close'].iloc[-1]
                        current_value += position['shares'] * price
                
                # Mettre à jour le capital
                current_capital = current_value
                
                # Vendre toutes les positions
                current_portfolio = {}
                
                # Acheter de nouvelles positions
                position_size = current_capital / len(top_stocks)
                
                for symbol in top_stocks:
                    if symbol in filtered_data and not filtered_data[symbol].empty:
                        price = filtered_data[symbol]['Close'].iloc[-1]
                        shares = position_size / price
                        current_portfolio[symbol] = {'shares': shares, 'entry_price': price}
            
            last_rebalance_date = rebalance_date
            
        except Exception as e:
            print(f"Erreur lors du rééquilibrage à {rebalance_date}: {e}")
            continue
    
    # Calculer la valeur du portefeuille pour chaque jour
    for date in results.index:
        if date <= start_date:
            continue
        
        portfolio_value = 0
        
        for symbol, position in current_portfolio.items():
            if symbol in price_data:
                df = price_data[symbol]
                mask = (df.index <= date)
                if mask.any():
                    latest_data = df[mask].iloc[-1]
                    if 'Close' in latest_data:
                        price = latest_data['Close']
                        portfolio_value += position['shares'] * price
        
        results.loc[date, 'portfolio_value'] = portfolio_value
    
    # Calculer les métriques de performance
    results['daily_returns'] = results['portfolio_value'].pct_change()
    results['cumulative_returns'] = (1 + results['daily_returns']).cumprod() - 1
    
    # Ignorer les premiers jours sans données
    results = results.fillna(method='ffill')
    
    return results
