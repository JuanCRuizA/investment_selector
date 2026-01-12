"""
Portfolio Calculations Module - Dynamic calculations for analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class PortfolioCalculations:
    """
    Handles dynamic calculations for portfolio analysis.
    Combines pre-computed data with real-time calculations based on user inputs.
    """
    
    def __init__(self, equity_curves: pd.DataFrame, backtest_summary: pd.DataFrame):
        """
        Initialize calculations with data.
        
        Args:
            equity_curves: Daily equity values for all profiles
            backtest_summary: Summary metrics from pipeline
        """
        self.equity_curves = equity_curves
        self.backtest_summary = backtest_summary
    
    def get_equity_curve_for_profile(self, profile: str) -> pd.DataFrame:
        """
        Get equity curve for specific profile.
        
        Args:
            profile: Profile name
            
        Returns:
            DataFrame with fecha, equity_portafolio, equity_benchmark
        """
        df = self.equity_curves[self.equity_curves['perfil'] == profile].copy()
        df = df.sort_values('fecha')
        return df
    
    def calculate_cumulative_returns(
        self,
        profile: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Calculate cumulative returns for portfolio vs benchmark.
        
        Args:
            profile: Profile name
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with cumulative returns
        """
        df = self.get_equity_curve_for_profile(profile)
        
        if start_date:
            df = df[df['fecha'] >= start_date]
        if end_date:
            df = df[df['fecha'] <= end_date]
        
        if df.empty:
            return df
        
        # Calculate cumulative returns from starting equity
        initial_portfolio = df['equity_portafolio'].iloc[0]
        initial_benchmark = df['equity_benchmark'].iloc[0]
        
        df['return_portfolio_pct'] = (df['equity_portafolio'] / initial_portfolio - 1) * 100
        df['return_benchmark_pct'] = (df['equity_benchmark'] / initial_benchmark - 1) * 100
        df['excess_return_pct'] = df['return_portfolio_pct'] - df['return_benchmark_pct']
        
        return df
    
    def calculate_monthly_returns(self, profile: str) -> pd.DataFrame:
        """
        Calculate monthly returns for a profile.
        
        Args:
            profile: Profile name
            
        Returns:
            DataFrame with monthly returns
        """
        df = self.get_equity_curve_for_profile(profile)
        
        if df.empty:
            return df
        
        df = df.set_index('fecha')
        
        # Resample to monthly and calculate returns
        monthly = df.resample('M').last()
        
        monthly['return_portfolio'] = monthly['equity_portafolio'].pct_change()
        monthly['return_benchmark'] = monthly['equity_benchmark'].pct_change()
        monthly['excess_return'] = monthly['return_portfolio'] - monthly['return_benchmark']
        
        monthly = monthly.reset_index()
        monthly['year'] = monthly['fecha'].dt.year
        monthly['month'] = monthly['fecha'].dt.month
        monthly['month_name'] = monthly['fecha'].dt.strftime('%b')
        
        return monthly.dropna()
    
    def calculate_annual_returns(self, profile: str) -> pd.DataFrame:
        """
        Calculate annual returns for a profile.
        
        Args:
            profile: Profile name
            
        Returns:
            DataFrame with annual returns
        """
        df = self.get_equity_curve_for_profile(profile)
        
        if df.empty:
            return df
        
        df = df.set_index('fecha')
        
        # Get first and last values per year
        yearly_first = df.resample('Y').first()
        yearly_last = df.resample('Y').last()
        
        annual = pd.DataFrame({
            'year': yearly_last.index.year,
            'equity_start_portfolio': yearly_first['equity_portafolio'].values,
            'equity_end_portfolio': yearly_last['equity_portafolio'].values,
            'equity_start_benchmark': yearly_first['equity_benchmark'].values,
            'equity_end_benchmark': yearly_last['equity_benchmark'].values,
        })
        
        annual['return_portfolio'] = (
            annual['equity_end_portfolio'] / annual['equity_start_portfolio'] - 1
        )
        annual['return_benchmark'] = (
            annual['equity_end_benchmark'] / annual['equity_start_benchmark'] - 1
        )
        annual['excess_return'] = annual['return_portfolio'] - annual['return_benchmark']
        
        return annual
    
    def calculate_drawdown(self, profile: str) -> pd.DataFrame:
        """
        Calculate drawdown series for a profile.
        
        Args:
            profile: Profile name
            
        Returns:
            DataFrame with drawdown values
        """
        df = self.get_equity_curve_for_profile(profile)
        
        if df.empty:
            return df
        
        # Calculate running maximum
        df['peak_portfolio'] = df['equity_portafolio'].cummax()
        df['peak_benchmark'] = df['equity_benchmark'].cummax()
        
        # Calculate drawdown
        df['drawdown_portfolio'] = (
            df['equity_portafolio'] / df['peak_portfolio'] - 1
        ) * 100
        df['drawdown_benchmark'] = (
            df['equity_benchmark'] / df['peak_benchmark'] - 1
        ) * 100
        
        return df
    
    def calculate_rolling_metrics(
        self,
        profile: str,
        window_days: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling Sharpe ratio and volatility.
        
        Args:
            profile: Profile name
            window_days: Rolling window in trading days
            
        Returns:
            DataFrame with rolling metrics
        """
        df = self.get_equity_curve_for_profile(profile)
        
        if df.empty or len(df) < window_days:
            return df
        
        df = df.set_index('fecha').copy()
        
        # Calculate daily returns
        df['daily_return_portfolio'] = df['equity_portafolio'].pct_change()
        df['daily_return_benchmark'] = df['equity_benchmark'].pct_change()
        
        # Rolling metrics
        risk_free_daily = 0.05 / 252  # 5% annual risk-free rate
        
        df['rolling_vol_portfolio'] = (
            df['daily_return_portfolio'].rolling(window_days).std() * np.sqrt(252)
        )
        df['rolling_vol_benchmark'] = (
            df['daily_return_benchmark'].rolling(window_days).std() * np.sqrt(252)
        )
        
        rolling_return = df['daily_return_portfolio'].rolling(window_days).mean() * 252
        rolling_vol = df['rolling_vol_portfolio']
        
        df['rolling_sharpe'] = (rolling_return - 0.05) / rolling_vol
        
        return df.reset_index()
    
    def get_metrics_comparison(self, profile: str) -> Dict:
        """
        Get formatted metrics comparison for display.
        
        Args:
            profile: Profile name
            
        Returns:
            Dictionary with formatted metrics
        """
        summary = self.backtest_summary[
            self.backtest_summary['perfil'] == profile
        ]
        
        if summary.empty:
            return {}
        
        row = summary.iloc[0]
        
        return {
            'portfolio': {
                'Return Total': f"{row['Return_Portfolio']:.2%}",
                'Return Anualizado': f"{row.get('Return_Annual_Portfolio', row['Return_Portfolio']):.2%}",
                'Volatilidad': f"{row['Volatility_Portfolio']:.2%}",
                'Sharpe Ratio': f"{row['Sharpe_Portfolio']:.2f}",
                'Max Drawdown': f"{row['MaxDD_Portfolio']:.2%}",
            },
            'benchmark': {
                'Return Total': f"{row['Return_Benchmark']:.2%}",
                'Return Anualizado': f"{row.get('Return_Annual_Benchmark', row['Return_Benchmark']):.2%}",
                'Volatilidad': f"{row['Volatility_Benchmark']:.2%}",
                'Sharpe Ratio': f"{row['Sharpe_Benchmark']:.2f}",
                'Max Drawdown': f"{row['MaxDD_Benchmark']:.2%}",
            },
            'comparison': {
                'Excess Return': f"{row['Excess_Return']:.2%}",
                'Winner': '🏆 Portafolio' if row['Excess_Return'] > 0 else '📊 Benchmark',
            }
        }
    
    def calculate_investment_projection(
        self,
        profile: str,
        initial_investment: float,
        years: int = 5
    ) -> pd.DataFrame:
        """
        Project investment value based on historical returns.
        
        Args:
            profile: Profile name
            initial_investment: Starting amount
            years: Projection years
            
        Returns:
            DataFrame with projected values
        """
        summary = self.backtest_summary[
            self.backtest_summary['perfil'] == profile
        ]
        
        if summary.empty:
            return pd.DataFrame()
        
        row = summary.iloc[0]
        annual_return = row.get('Return_Annual_Portfolio', row['Return_Portfolio'] / 2)
        annual_vol = row['Volatility_Portfolio']
        
        projections = []
        for year in range(years + 1):
            expected = initial_investment * ((1 + annual_return) ** year)
            # Simple confidence intervals
            lower = initial_investment * ((1 + annual_return - annual_vol) ** year)
            upper = initial_investment * ((1 + annual_return + annual_vol) ** year)
            
            projections.append({
                'year': year,
                'expected': expected,
                'lower_bound': max(0, lower),
                'upper_bound': upper,
            })
        
        return pd.DataFrame(projections)
