"""
Module pour visualiser les résultats de l'analyse momentum
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os


class MomentumPlotter:
    """
    Classe pour visualiser les résultats de l'analyse momentum
    """
    
    def __init__(self, output_dir: str = "data/plots"):
        """
        Initialise le MomentumPlotter
        
        Args:
            output_dir: Répertoire où sauvegarder les graphiques
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration de style pour seaborn
        sns.set_theme(style="whitegrid")
        
    def plot_momentum_scores(self, 
                           momentum_df: pd.DataFrame, 
                           n_stocks: int = 20,
                           title: str = "Top Momentum Scores",
                           save_path: Optional[str] = None):
        """
        Visualise les scores de momentum pour les N meilleures actions
        
        Args:
            momentum_df: DataFrame avec les scores de momentum
            n_stocks: Nombre d'actions à afficher
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique (None pour ne pas sauvegarder)
        """
        # Sélectionner les N meilleures actions
        top_stocks = momentum_df.head(n_stocks)
        
        plt.figure(figsize=(12, 8))
        
        # Créer un barplot horizontal
        ax = sns.barplot(x='momentum_score', y=top_stocks.index, data=top_stocks, palette='viridis')
        
        # Ajouter un titre et des labels
        plt.title(title, fontsize=16)
        plt.xlabel('Score de Momentum', fontsize=12)
        plt.ylabel('Symbole', fontsize=12)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(top_stocks['momentum_score']):
            ax.text(max(v + 0.01, 0.01), i, f"{v:.4f}", va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé à {save_path}")
            
        return plt.gcf()
    
    def plot_momentum_percentiles(self, 
                                momentum_df: pd.DataFrame, 
                                n_stocks: int = 20,
                                title: str = "Top Momentum Percentiles",
                                save_path: Optional[str] = None):
        """
        Visualise les percentiles de momentum pour les N meilleures actions
        
        Args:
            momentum_df: DataFrame avec les percentiles de momentum
            n_stocks: Nombre d'actions à afficher
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique (None pour ne pas sauvegarder)
        """
        # Sélectionner les N meilleures actions
        top_stocks = momentum_df.head(n_stocks)
        
        plt.figure(figsize=(12, 8))
        
        # Créer un barplot horizontal
        ax = sns.barplot(x='momentum_percentile', y=top_stocks.index, data=top_stocks, palette='viridis')
        
        # Ajouter un titre et des labels
        plt.title(title, fontsize=16)
        plt.xlabel('Percentile de Momentum', fontsize=12)
        plt.ylabel('Symbole', fontsize=12)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(top_stocks['momentum_percentile']):
            ax.text(v + 1, i, f"{v:.2f}", va='center', fontsize=10)
        
        # Limiter l'axe x à 100
        plt.xlim(0, 105)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé à {save_path}")
            
        return plt.gcf()
    
    def plot_hqm_scores(self, 
                       hqm_df: pd.DataFrame, 
                       n_stocks: int = 20,
                       title: str = "Top HQM Scores",
                       save_path: Optional[str] = None):
        """
        Visualise les scores HQM pour les N meilleures actions
        
        Args:
            hqm_df: DataFrame avec les scores HQM
            n_stocks: Nombre d'actions à afficher
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique (None pour ne pas sauvegarder)
        """
        # Sélectionner les N meilleures actions
        top_stocks = hqm_df.head(n_stocks)
        
        plt.figure(figsize=(12, 8))
        
        # Créer un barplot horizontal
        ax = sns.barplot(x='hqm_score', y=top_stocks.index, data=top_stocks, palette='viridis')
        
        # Ajouter un titre et des labels
        plt.title(title, fontsize=16)
        plt.xlabel('Score HQM (High Quality Momentum)', fontsize=12)
        plt.ylabel('Symbole', fontsize=12)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(top_stocks['hqm_score']):
            ax.text(v + 1, i, f"{v:.2f}", va='center', fontsize=10)
        
        # Limiter l'axe x à 100
        plt.xlim(0, 105)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé à {save_path}")
            
        return plt.gcf()
    
    def plot_returns_heatmap(self, 
                           returns_df: pd.DataFrame,
                           periods: List[int] = [30, 90, 180, 365],
                           n_stocks: int = 20,
                           title: str = "Returns Heatmap",
                           save_path: Optional[str] = None):
        """
        Crée une heatmap des rendements pour les N meilleures actions
        
        Args:
            returns_df: DataFrame avec les rendements
            periods: Liste des périodes en jours
            n_stocks: Nombre d'actions à afficher
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique (None pour ne pas sauvegarder)
        """
        # Sélectionner les N meilleures actions si un score est disponible
        if 'hqm_score' in returns_df.columns:
            top_stocks = returns_df.sort_values('hqm_score', ascending=False).head(n_stocks)
        else:
            top_stocks = returns_df.head(n_stocks)
        
        # Sélectionner seulement les colonnes de rendement
        return_columns = [f'return_{period}d' for period in periods if f'return_{period}d' in top_stocks.columns]
        
        if not return_columns:
            print("Aucune colonne de rendement disponible")
            return None
        
        # Créer un DataFrame pivot pour la heatmap
        heatmap_data = top_stocks[return_columns].copy()
        
        # Renommer les colonnes pour l'affichage
        column_mapping = {f'return_{period}d': f'{period} jours' for period in periods}
        heatmap_data = heatmap_data.rename(columns=column_mapping)
        
        plt.figure(figsize=(12, 10))
        
        # Créer la heatmap
        ax = sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt=".2%", linewidths=.5, center=0)
        
        # Ajouter un titre
        plt.title(title, fontsize=16)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé à {save_path}")
            
        return plt.gcf()
    
    def plot_portfolio_allocation(self, 
                                portfolio_df: pd.DataFrame,
                                title: str = "Portfolio Allocation",
                                save_path: Optional[str] = None):
        """
        Visualise l'allocation du portefeuille
        
        Args:
            portfolio_df: DataFrame avec l'allocation du portefeuille
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique (None pour ne pas sauvegarder)
        """
        plt.figure(figsize=(12, 8))
        
        # Créer un barplot horizontal
        ax = sns.barplot(x='position_size', y=portfolio_df.index, data=portfolio_df, palette='viridis')
        
        # Ajouter un titre et des labels
        plt.title(title, fontsize=16)
        plt.xlabel('Taille de Position ($)', fontsize=12)
        plt.ylabel('Symbole', fontsize=12)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(portfolio_df['position_size']):
            ax.text(v + 100, i, f"${v:.2f}", va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé à {save_path}")
            
        return plt.gcf()
    
    def plot_price_history(self,
                         price_data: Dict[str, pd.DataFrame],
                         symbols: List[str],
                         window: int = 252,  # 1 an de trading
                         title: str = "Historical Price Performance",
                         save_path: Optional[str] = None):
        """
        Visualise l'historique des prix normalisés pour comparer la performance
        
        Args:
            price_data: Dictionnaire de DataFrames avec les données de prix
            symbols: Liste des symboles à afficher
            window: Fenêtre temporelle à afficher en jours
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique (None pour ne pas sauvegarder)
        """
        plt.figure(figsize=(14, 8))
        
        for symbol in symbols:
            if symbol in price_data and 'Close' in price_data[symbol].columns:
                df = price_data[symbol].copy()
                
                # Limiter à la fenêtre temporelle spécifiée
                if len(df) > window:
                    df = df.iloc[-window:]
                
                # Normaliser les prix (base 100)
                normalized_prices = df['Close'] / df['Close'].iloc[0] * 100
                
                # Tracer la courbe
                plt.plot(normalized_prices.index, normalized_prices, label=symbol, linewidth=2)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Prix normalisé (base 100)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique si un chemin est spécifié
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé à {save_path}")
            
        return plt.gcf()
