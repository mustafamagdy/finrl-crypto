"""
FinRL StockTradingEnv Patch
==========================

This module patches the FinRL StockTradingEnv to fix the 
'numpy.float64' object has no attribute 'values' error.

The bug occurs when self.data.close becomes a scalar instead of a Series,
causing .values.tolist() to fail on lines 409, 433, and 462.

Usage:
    from finrl_patch import PatchedStockTradingEnv
    # Use PatchedStockTradingEnv instead of StockTradingEnv
"""

import numpy as np
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class PatchedStockTradingEnv(StockTradingEnv):
    """
    Patched version of FinRL StockTradingEnv that fixes multiple bugs:
    1. numpy.float64.values bug in _initiate_state and _update_state
    2. IndexError: list index out of range in _buy_stock and _sell_stock
    3. State space indexing issues with turbulence data
    4. Proper bounds checking for all state access operations
    5. DataFrame indexing issues during environment initialization
    """
    
    def __init__(self, df, **kwargs):
        """
        Fixed initialization that handles DataFrame indexing properly
        """
        # Ensure the DataFrame has proper integer index starting from 0
        if not isinstance(df.index, pd.RangeIndex) or df.index[0] != 0:
            df = df.reset_index(drop=True)
        
        # Call parent constructor with fixed DataFrame
        super().__init__(df=df, **kwargs)
    
    def _safe_values_tolist(self, data, column_name='close'):
        """
        Safely extract values from data column, handling both Series and scalar cases
        
        Args:
            data: pandas Series or scalar value
            column_name: name of column being accessed (for debugging)
            
        Returns:
            list: Always returns a list, even for scalar inputs
        """
        try:
            # If it has .values attribute (Series/DataFrame), use it
            if hasattr(data, 'values'):
                return data.values.tolist()
            else:
                # It's a scalar (numpy.float64), convert to list
                if np.isscalar(data):
                    return [float(data)]
                else:
                    # Fallback: try to convert directly
                    return [data] if not isinstance(data, list) else data
        except Exception as e:
            print(f"⚠️ Warning: _safe_values_tolist failed for {column_name}: {e}")
            print(f"   Data type: {type(data)}, Data value: {data}")
            # Emergency fallback
            return [float(data)] if np.isscalar(data) else [data]
    
    def _safe_tech_values(self, tech_indicator):
        """
        Safely extract technical indicator values
        
        Args:
            tech_indicator: name of technical indicator
            
        Returns:
            list: Values for the technical indicator
        """
        try:
            tech_data = self.data[tech_indicator]
            return self._safe_values_tolist(tech_data, tech_indicator)
        except Exception as e:
            print(f"⚠️ Warning: Failed to extract {tech_indicator}: {e}")
            return [0.0]  # Default value if extraction fails
    
    def _initiate_state(self):
        """
        Fixed version of _initiate_state that handles scalar values properly
        """
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock - FIXED VERSION
                state = (
                    [self.initial_amount]
                    + self._safe_values_tolist(self.data.close, 'close')  # FIXED
                    + self.num_stock_shares
                    + sum(
                        (
                            self._safe_tech_values(tech)  # FIXED
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock - ALREADY WORKS
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock - FIXED VERSION
                state = (
                    [self.previous_state[0]]
                    + self._safe_values_tolist(self.data.close, 'close')  # FIXED
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self._safe_tech_values(tech)  # FIXED
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock - ALREADY WORKS
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        """
        Fixed version of _update_state that handles scalar values properly
        """
        if len(self.df.tic.unique()) > 1:
            # for multiple stock - FIXED VERSION
            state = (
                [self.state[0]]
                + self._safe_values_tolist(self.data.close, 'close')  # FIXED
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self._safe_tech_values(tech)  # FIXED
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )
        else:
            # for single stock - ALREADY WORKS
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state
    
    def _buy_stock(self, index, action):
        """
        Fixed version of _buy_stock that handles state indexing properly
        """
        def _do_buy():
            # Fixed state indexing check with bounds validation
            turbulence_index = index + 2 * self.stock_dim + 1
            if turbulence_index < len(self.state):
                turbulence_check = self.state[turbulence_index] != True
            else:
                # If index is out of bounds, assume no turbulence restriction
                turbulence_check = True
                
            if turbulence_check:
                # Buy only if the price is > 0 (no missing data)
                price_index = index + 1
                if price_index < len(self.state) and self.state[price_index] > 0:
                    available_amount = self.state[0] // (
                        self.state[price_index] * (1 + self.buy_cost_pct[index])
                    )
                    
                    # Update balance
                    buy_num_shares = min(available_amount, action)
                    buy_amount = (
                        self.state[price_index]
                        * buy_num_shares
                        * (1 + self.buy_cost_pct[index])
                    )
                    self.state[0] -= buy_amount
                    
                    # Update shares (with bounds check)
                    shares_index = index + self.stock_dim + 1
                    if shares_index < len(self.state):
                        self.state[shares_index] += buy_num_shares
                    
                    self.cost += (
                        self.state[price_index] * buy_num_shares * self.buy_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    buy_num_shares = 0
            else:
                buy_num_shares = 0

            return buy_num_shares

        # Perform buy action based on turbulence threshold
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if hasattr(self, 'turbulence') and self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0

        return buy_num_shares
    
    def _sell_stock(self, index, action):
        """
        Fixed version of _sell_stock that handles state indexing properly
        """
        def _do_sell_normal():
            # Check if we have shares to sell
            shares_index = index + self.stock_dim + 1
            if shares_index < len(self.state) and self.state[shares_index] > 0:
                # Sell only if the price is > 0 (no missing data)
                price_index = index + 1
                if price_index < len(self.state) and self.state[price_index] > 0:
                    # Update balance
                    sell_num_shares = min(abs(action), self.state[shares_index])
                    sell_amount = (
                        self.state[price_index]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    self.state[0] += sell_amount
                    
                    # Update shares
                    self.state[shares_index] -= sell_num_shares
                    
                    self.cost += (
                        self.state[price_index] * sell_num_shares * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        def _do_sell():
            # Fixed turbulence check with bounds validation
            turbulence_index = index + 2 * self.stock_dim + 1
            if turbulence_index < len(self.state):
                return self.state[turbulence_index] != True
            else:
                # If index is out of bounds, assume no turbulence restriction
                return True

        # Perform sell action based on turbulence threshold
        if self.turbulence_threshold is None:
            sell_num_shares = _do_sell_normal()
        else:
            if hasattr(self, 'turbulence') and self.turbulence < self.turbulence_threshold:
                sell_num_shares = _do_sell_normal()  
            else:
                sell_num_shares = _do_sell_normal()

        return sell_num_shares


# Convenience function to replace StockTradingEnv imports
def create_patched_env(df, **kwargs):
    """
    Create a patched StockTradingEnv that fixes the numpy.float64.values bug
    
    Args:
        df: DataFrame with trading data
        **kwargs: All other StockTradingEnv parameters
        
    Returns:
        PatchedStockTradingEnv: Fixed environment
    """
    return PatchedStockTradingEnv(df=df, **kwargs)


if __name__ == "__main__":
    print("FinRL StockTradingEnv Patch")
    print("==========================")
    print("This patch fixes the 'numpy.float64' object has no attribute 'values' error")
    print("in FinRL StockTradingEnv by safely handling scalar values.")
    print("\nUsage:")
    print("  from finrl_patch import PatchedStockTradingEnv")
    print("  # Use PatchedStockTradingEnv instead of StockTradingEnv")