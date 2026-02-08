"""
Unit tests for the Inventory Optimization Tool.
"""
import pytest
import numpy as np
from src.optimization.optimizer import InventoryOptimizer


class TestEconomicOrderQuantity:
    """Tests for EOQ calculations."""

    def test_eoq_basic_calculation(self):
        """Test basic EOQ formula."""
        # Known values: D=1000, S=50, H=2
        # EOQ = sqrt(2*1000*50/2) = sqrt(50000) = 223.6
        eoq = InventoryOptimizer.calculate_eoq(
            annual_demand=1000,
            order_cost=50,
            holding_cost=2
        )
        assert abs(eoq - 223.6) < 0.1

    def test_eoq_zero_holding_cost(self):
        """Test EOQ returns 0 for zero holding cost."""
        eoq = InventoryOptimizer.calculate_eoq(
            annual_demand=1000,
            order_cost=50,
            holding_cost=0
        )
        assert eoq == 0

    def test_eoq_negative_holding_cost(self):
        """Test EOQ returns 0 for negative holding cost."""
        eoq = InventoryOptimizer.calculate_eoq(
            annual_demand=1000,
            order_cost=50,
            holding_cost=-5
        )
        assert eoq == 0

    def test_eoq_large_values(self):
        """Test EOQ with large industrial values."""
        eoq = InventoryOptimizer.calculate_eoq(
            annual_demand=100000,
            order_cost=500,
            holding_cost=25
        )
        assert eoq > 0
        assert not np.isnan(eoq)


class TestSafetyStock:
    """Tests for safety stock calculations."""

    def test_safety_stock_95_service_level(self):
        """Test safety stock at 95% service level."""
        ss = InventoryOptimizer.calculate_safety_stock(
            avg_lead_time=10,
            std_dev_demand=20,
            service_level=0.95
        )
        # z-score for 95% is 1.645
        expected = 1.645 * np.sqrt(10) * 20
        assert abs(ss - expected) < 1

    def test_safety_stock_99_service_level(self):
        """Test safety stock at 99% service level."""
        ss = InventoryOptimizer.calculate_safety_stock(
            avg_lead_time=5,
            std_dev_demand=15,
            service_level=0.99
        )
        # Higher service level = more safety stock
        ss_95 = InventoryOptimizer.calculate_safety_stock(5, 15, 0.95)
        assert ss > ss_95

    def test_safety_stock_zero_variability(self):
        """Test safety stock with zero demand variability."""
        ss = InventoryOptimizer.calculate_safety_stock(
            avg_lead_time=10,
            std_dev_demand=0,
            service_level=0.95
        )
        assert ss == 0


class TestReorderPoint:
    """Tests for reorder point calculations."""

    def test_reorder_point_basic(self):
        """Test basic reorder point calculation."""
        rop = InventoryOptimizer.calculate_reorder_point(
            avg_demand=100,
            avg_lead_time=5,
            safety_stock=50
        )
        # ROP = (100 * 5) + 50 = 550
        assert rop == 550

    def test_reorder_point_no_safety_stock(self):
        """Test reorder point without safety stock."""
        rop = InventoryOptimizer.calculate_reorder_point(
            avg_demand=200,
            avg_lead_time=3,
            safety_stock=0
        )
        assert rop == 600


class TestSKUOptimization:
    """Tests for complete SKU optimization."""

    def test_optimize_sku_complete(self):
        """Test complete SKU optimization."""
        sku_data = {
            'annual_demand': 12000,
            'order_cost': 100,
            'holding_cost': 5,
            'avg_lead_time': 7,
            'std_dev_demand': 30,
            'avg_demand': 33  # 12000/365 ≈ 33
        }
        
        result = InventoryOptimizer.optimize_sku(sku_data)
        
        assert 'eoq' in result
        assert 'safety_stock' in result
        assert 'reorder_point' in result
        assert result['eoq'] > 0
        assert result['safety_stock'] > 0
        assert result['reorder_point'] > 0

    def test_optimize_sku_output_types(self):
        """Test that output values are proper floats."""
        sku_data = {
            'annual_demand': 5000,
            'order_cost': 75,
            'holding_cost': 3,
            'avg_lead_time': 14,
            'std_dev_demand': 25,
            'avg_demand': 14
        }
        
        result = InventoryOptimizer.optimize_sku(sku_data)
        
        assert isinstance(result['eoq'], float)
        assert isinstance(result['safety_stock'], float)
        assert isinstance(result['reorder_point'], float)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_very_small_values(self):
        """Test with very small demand values."""
        eoq = InventoryOptimizer.calculate_eoq(1, 1, 1)
        assert eoq > 0
        assert not np.isinf(eoq)

    def test_decimal_precision(self):
        """Test decimal precision in calculations."""
        eoq = InventoryOptimizer.calculate_eoq(
            annual_demand=1234.56,
            order_cost=78.90,
            holding_cost=1.23
        )
        assert isinstance(eoq, (int, float))
        assert eoq > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
