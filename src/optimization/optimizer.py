import numpy as np
import pandas as pd
from scipy.stats import norm

class InventoryOptimizer:
    @staticmethod
    def calculate_eoq(annual_demand, order_cost, holding_cost):
        """Calculates Economic Order Quantity (EOQ)."""
        if holding_cost <= 0:
            return 0
        return np.sqrt((2 * annual_demand * order_cost) / holding_cost)

    @staticmethod
    def calculate_safety_stock(avg_lead_time, std_dev_demand, service_level=0.95):
        """Calculates safety stock based on demand variability."""
        z_score = norm.ppf(service_level)
        return z_score * np.sqrt(avg_lead_time) * std_dev_demand

    @staticmethod
    def calculate_reorder_point(avg_demand, avg_lead_time, safety_stock):
        """Calculates the Reorder Point (ROP)."""
        return (avg_demand * avg_lead_time) + safety_stock

    @classmethod
    def optimize_sku(cls, sku_data):
        """Optimizes inventory parameters for a single SKU."""
        demand = sku_data.get('annual_demand', 0)
        order_cost = sku_data.get('order_cost', 0)
        holding_cost = sku_data.get('holding_cost', 0)
        
        eoq = cls.calculate_eoq(demand, order_cost, holding_cost)
        ss = cls.calculate_safety_stock(sku_data['avg_lead_time'], sku_data['std_dev_demand'])
        rop = cls.calculate_reorder_point(sku_data['avg_demand'], sku_data['avg_lead_time'], ss)
        
        return {
            "eoq": float(eoq),
            "safety_stock": float(ss),
            "reorder_point": float(rop)
        }
