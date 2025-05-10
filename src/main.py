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


def main():
    """
    Fonction principale
    """
    # Analyser les arguments
    args = parse_arguments()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialiser les classes
    data_fetcher = DataFetcher(cache_dir='data/raw')
    momentum_calculator = MomentumCalculator()
    plotter = MomentumPlotter(output_dir=os.path.join(args.output_dir, 'plots'))
    
    # Récupérer les symboles en fonction de l'indice
    if args.index.lower() == 'sp500':
        symbols = data_fetcher.get_sp500_symbols()
    else:
        raise ValueError(f"Indice non pris en charge: {args.index}")
    
    print(f"Récupération des données pour {len(symbols)} symboles...")
    
    # Appliquer les filtres de capitalisation boursière
    if args.min_market_cap is not None or args.max_market_cap is not None:
        min_cap = args.min_market_cap * 1e6 if args.min_market_cap else None
        max_cap = args.max_market_cap * 1e6 if args.max_market_cap else None
        
        symbols = filter_stocks_by_market_cap(symbols, min_cap, max_cap, data_fetcher)
        
        print(f"Après filtrage par capitalisation boursière: {len(symbols)} symboles")
    
    # Récupérer les données historiques
    # Utiliser une année de données
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"Récupération des données historiques de {start_date} à {end_date}...")
    
    historical_data = data_fetcher.get_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=args.use_cache
    )
    
    # Filtrer les actions avec suffisamment de données
    valid_symbols = [symbol for symbol, data in historical_data.items() 
                   if len(data) >= args.lookback_days]
    
    print(f"Après filtrage pour données suffisantes: {len(valid_symbols)} symboles")
    
    # Calculer les scores de momentum
    print(f"Calcul des scores de momentum...")
    
    momentum_df = momentum_calculator.calculate_momentum_percentile(
        {symbol: historical_data[symbol] for symbol in valid_symbols},
        lookback_days=args.lookback_days
    )
    
    # Calculer les scores HQM en utilisant différentes périodes
    print(f"Calcul des scores HQM...")
    
    hqm_df = momentum_calculator.calculate_hqm_score(
        {symbol: historical_data[symbol] for symbol in valid_symbols},
        periods=[30, 90, 180, 365]  # 1 mois, 3 mois, 6 mois, 1 an
    )
    
    # Fusionner les scores
    combined_df = momentum_df.copy()
    if 'hqm_score' in hqm_df.columns:
        combined_df['hqm_score'] = hqm_df['hqm_score']
    
    # Calculer la volatilité pour l'allocation basée sur le risque
    volatility_dict = None
    if args.risk_based:
        print(f"Calcul de la volatilité pour l'allocation basée sur le risque...")
        volatility_dict = calculate_volatility_dict(
            {symbol: historical_data[symbol] for symbol in valid_symbols},
            window=20  # ATR sur 20 jours
        )
    
    # Calculer l'allocation de portefeuille
    print(f"Calcul de l'allocation de portefeuille...")
    
    portfolio_df = momentum_calculator.calculate_optimal_portfolio(
        combined_df,
        portfolio_size=args.portfolio_size,
        n_stocks=args.n_stocks,
        risk_based_allocation=args.risk_based,
        volatility_data=volatility_dict
    )
    
    # Sélectionner les top N actions
    top_stocks = portfolio_df.head(args.n_stocks).index.tolist()
    
    # Visualisation des résultats
    print(f"Génération des visualisations...")
    
    # Plot des scores de momentum
    momentum_plot = plotter.plot_momentum_scores(
        momentum_df,
        n_stocks=min(20, len(momentum_df)),
        title=f"Top Momentum Scores - {args.index.upper()}",
        save_path=os.path.join(args.output_dir, 'plots', 'momentum_scores.png')
    )
    
    # Plot des percentiles de momentum
    percentile_plot = plotter.plot_momentum_percentiles(
        momentum_df,
        n_stocks=min(20, len(momentum_df)),
        title=f"Top Momentum Percentiles - {args.index.upper()}",
        save_path=os.path.join(args.output_dir, 'plots', 'momentum_percentiles.png')
    )
    
    # Plot des scores HQM si disponibles
    if 'hqm_score' in combined_df.columns:
        hqm_plot = plotter.plot_hqm_scores(
            combined_df,
            n_stocks=min(20, len(combined_df)),
            title=f"Top HQM Scores - {args.index.upper()}",
            save_path=os.path.join(args.output_dir, 'plots', 'hqm_scores.png')
        )
    
    # Plot de l'allocation du portefeuille
    portfolio_plot = plotter.plot_portfolio_allocation(
        portfolio_df.head(min(20, len(portfolio_df))),
        title=f"Portfolio Allocation - {args.index.upper()}",
        save_path=os.path.join(args.output_dir, 'plots', 'portfolio_allocation.png')
    )
    
    # Plot des performances historiques des top stocks
    performance_plot = plotter.plot_price_history(
        {symbol: historical_data[symbol] for symbol in top_stocks[:10]},  # Top 10 pour la lisibilité
        symbols=top_stocks[:10],
        window=252,  # 1 an de trading
        title=f"Historical Price Performance - Top 10 Momentum Stocks",
        save_path=os.path.join(args.output_dir, 'plots', 'price_performance.png')
    )
    
    # Export des résultats en Excel
    print(f"Export des résultats...")
    
    # Exporter les scores de momentum
    export_to_excel(
        momentum_df,
        file_path=os.path.join(args.output_dir, 'momentum_scores.xlsx'),
        sheet_name='Momentum Scores',
        format_dict={
            'momentum_score': 'number',
            'momentum_percentile': 'number'
        }
    )
    
    # Exporter l'allocation de portefeuille
    export_to_excel(
        portfolio_df,
        file_path=os.path.join(args.output_dir, 'portfolio_allocation.xlsx'),
        sheet_name='Portfolio Allocation',
        format_dict={
            'momentum_score': 'number',
            'momentum_percentile': 'number',
            'hqm_score': 'number',
            'position_size': 'dollar'
        }
    )
    
    print("Terminé! Les résultats ont été sauvegardés dans le répertoire:", args.output_dir)
    
    return top_stocks, portfolio_df, momentum_df, hqm_df


if __name__ == "__main__":
    main()
