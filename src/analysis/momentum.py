"""
Module pour calculer les scores de momentum des actions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import linregress
from scipy.stats import percentileofscore


class MomentumCalculator:
    """
    Classe pour calculer les scores de momentum des actions
    """
    
    def __init__(self):
        """
        Initialise le MomentumCalculator
        """
        pass
    
    def calculate_returns(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calcule les rendements sur différentes périodes
        
        Args:
            data: DataFrame des données historiques avec une colonne 'Close'
            periods: Liste des périodes (en jours) pour lesquelles calculer les rendements
            
        Returns:
            DataFrame avec les rendements pour chaque période
        """
        result = pd.DataFrame(index=data.index)
        close_prices = data['Close']
        
        for period in periods:
            result[f'return_{period}d'] = close_prices.pct_change(period)
            
        return result
    
    def calculate_momentum_regression(self, closes: np.ndarray, days: int = 90) -> float:
        """
        Calcule le momentum en utilisant la régression exponentielle
        
        Args:
            closes: Tableau des prix de clôture
            days: Nombre de jours pour le calcul
            
        Returns:
            Score de momentum (pente annualisée * R²)
        """
        if len(closes) < days:
            return np.nan
        
        # Utiliser les derniers 'days' jours
        closes = closes[-days:]
        
        # Calculer les rendements logarithmiques
        log_returns = np.log(closes)
        
        # Créer un array pour l'axe x (jours)
        x = np.arange(len(log_returns))
        
        # Effectuer la régression linéaire
        slope, _, r_value, _, _ = linregress(x, log_returns)
        
        # Annualiser la pente et multiplier par R²
        annualized_slope = ((1 + slope) ** 252) - 1  # 252 jours de trading par an
        momentum_score = annualized_slope * (r_value ** 2)
        
        return momentum_score
    
    def calculate_momentum_percentile(self, data: Dict[str, pd.DataFrame], lookback_days: int = 90) -> pd.DataFrame:
        """
        Calcule les scores de momentum en percentile pour plusieurs actions
        
        Args:
            data: Dictionnaire de DataFrames contenant les données historiques
            lookback_days: Nombre de jours pour le calcul du momentum
            
        Returns:
            DataFrame avec les scores de momentum en percentile
        """
        momentum_scores = {}
        
        for symbol, df in data.items():
            if 'Close' in df.columns and len(df) >= lookback_days:
                momentum_scores[symbol] = self.calculate_momentum_regression(df['Close'].values, lookback_days)
            
        # Créer un DataFrame avec les scores de momentum
        momentum_df = pd.DataFrame.from_dict(momentum_scores, orient='index', columns=['momentum_score'])
        
        # Calculer les percentiles
        valid_scores = momentum_df['momentum_score'].dropna()
        momentum_df['momentum_percentile'] = momentum_df['momentum_score'].apply(
            lambda x: percentileofscore(valid_scores, x) if not np.isnan(x) else np.nan
        )
        
        # Trier par score de momentum décroissant
        momentum_df = momentum_df.sort_values('momentum_score', ascending=False)
        
        return momentum_df
    
    def calculate_hqm_score(self, data: Dict[str, pd.DataFrame], 
                           periods: List[int] = [30, 90, 180, 365]) -> pd.DataFrame:
        """
        Calcule le score de momentum de haute qualité (HQM) basé sur plusieurs périodes
        
        Args:
            data: Dictionnaire de DataFrames contenant les données historiques
            periods: Liste des périodes (en jours) pour lesquelles calculer les rendements
            
        Returns:
            DataFrame avec les scores HQM
        """
        all_returns = {}
        
        for symbol, df in data.items():
            if 'Close' in df.columns:
                returns = self.calculate_returns(df, periods)
                all_returns[symbol] = returns.iloc[-1].to_dict()  # Prendre la dernière ligne
        
        # Créer un DataFrame avec tous les rendements
        returns_df = pd.DataFrame.from_dict(all_returns, orient='index')
        
        # Calculer les percentiles pour chaque période
        for period in periods:
            period_key = f'return_{period}d'
            if period_key in returns_df.columns:
                valid_returns = returns_df[period_key].dropna()
                returns_df[f'percentile_{period}d'] = returns_df[period_key].apply(
                    lambda x: percentileofscore(valid_returns, x) if not np.isnan(x) else np.nan
                )
        
        # Calculer le score HQM comme la moyenne des percentiles
        percentile_columns = [f'percentile_{period}d' for period in periods if f'percentile_{period}d' in returns_df.columns]
        
        if percentile_columns:
            returns_df['hqm_score'] = returns_df[percentile_columns].mean(axis=1)
        else:
            returns_df['hqm_score'] = np.nan
        
        # Trier par score HQM décroissant
        returns_df = returns_df.sort_values('hqm_score', ascending=False)
        
        return returns_df
    
    def get_top_momentum_stocks(self, 
                              momentum_df: pd.DataFrame, 
                              n: int = 50, 
                              min_percentile: Optional[float] = 70.0) -> List[str]:
        """
        Récupère les N actions avec le meilleur score de momentum
        
        Args:
            momentum_df: DataFrame avec les scores de momentum
            n: Nombre d'actions à retourner
            min_percentile: Percentile minimum requis (None pour ignorer)
            
        Returns:
            Liste des symboles des actions avec le meilleur momentum
        """
        if min_percentile is not None:
            filtered_df = momentum_df[momentum_df['momentum_percentile'] >= min_percentile]
        else:
            filtered_df = momentum_df
        
        # Retourner les N premiers symboles
        return filtered_df.index[:n].tolist()
    
    def calculate_optimal_portfolio(self, 
                                  momentum_df: pd.DataFrame, 
                                  portfolio_size: float, 
                                  n_stocks: int = 50,
                                  risk_based_allocation: bool = True,
                                  volatility_data: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calcule la répartition optimale du portefeuille basé sur le momentum
        
        Args:
            momentum_df: DataFrame avec les scores de momentum
            portfolio_size: Taille du portefeuille en unités monétaires
            n_stocks: Nombre d'actions à inclure dans le portefeuille
            risk_based_allocation: Utiliser l'allocation basée sur le risque
            volatility_data: Dictionnaire de la volatilité (ATR) par symbole
            
        Returns:
            DataFrame avec les allocations de portefeuille
        """
        # Sélectionner les N actions avec le meilleur score
        top_stocks = momentum_df.head(n_stocks)
        
        # Allocation équipondérée par défaut
        position_size = portfolio_size / len(top_stocks)
        
        # Créer le DataFrame de résultat
        portfolio_df = pd.DataFrame(index=top_stocks.index)
        portfolio_df['momentum_score'] = top_stocks['momentum_score']
        portfolio_df['momentum_percentile'] = top_stocks['momentum_percentile']
        
        if 'hqm_score' in top_stocks.columns:
            portfolio_df['hqm_score'] = top_stocks['hqm_score']
        
        if risk_based_allocation and volatility_data:
            # Calculer l'allocation basée sur le risque
            valid_stocks = [s for s in top_stocks.index if s in volatility_data]
            
            if valid_stocks:
                inverse_volatility = {s: 1/volatility_data[s] if volatility_data[s] > 0 else 0 for s in valid_stocks}
                total_inverse_vol = sum(inverse_volatility.values())
                
                if total_inverse_vol > 0:
                    for symbol in portfolio_df.index:
                        if symbol in inverse_volatility:
                            weight = inverse_volatility[symbol] / total_inverse_vol
                            portfolio_df.loc[symbol, 'position_size'] = portfolio_size * weight
                        else:
                            portfolio_df.loc[symbol, 'position_size'] = 0
                else:
                    portfolio_df['position_size'] = position_size
            else:
                portfolio_df['position_size'] = position_size
        else:
            # Allocation équipondérée
            portfolio_df['position_size'] = position_size
            
        return portfolio_df
