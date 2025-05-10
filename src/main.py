"""
Script principal pour exécuter la stratégie de momentum trading
"""

import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

from src.data.fetcher import DataFetcher
from src.analysis.momentum import MomentumCalculator
from src.visualization.plotter import MomentumPlotter
from src.utils.helpers import calculate_volatility_dict, export_to_excel, filter_stocks_by_market_cap


def parse_arguments():
    """
    Parse les arguments de la ligne de commande
    
    Returns:
        Arguments analysés
    """
    parser = argparse.ArgumentParser(description='Exécuter la stratégie de momentum trading')
    
    parser.add_argument('--index', type=str, default='sp500',
                      help='Indice à analyser (sp500, nasdaq100, etc.)')
    
    parser.add_argument('--n-stocks', type=int, default=50,
                      help='Nombre d\'actions à sélectionner')
    
    parser.add_argument('--min-market-cap', type=float, default=None,
                      help='Capitalisation boursière minimale (en millions de dollars)')
    
    parser.add_argument('--max-market-cap', type=float, default=None,
                      help='Capitalisation boursière maximale (en millions de dollars)')
    
    parser.add_argument('--lookback-days', type=int, default=90,
                      help='Période de lookback pour le calcul du momentum (en jours)')
    
    parser.add_argument('--portfolio-size', type=float, default=10000.0,
                      help='Taille du portefeuille en dollars')
    
    parser.add_argument('--risk-based', action='store_true',
                      help='Utiliser une allocation basée sur le risque')
    
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Répertoire de sortie pour les résultats')
    
    parser.add_argument('--use-cache', action='store_true',
                      help='Utiliser les données en cache')
    
    args = parser.parse_args()
    
    return args
