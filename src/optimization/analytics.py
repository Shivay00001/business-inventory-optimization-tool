"""
Inventory Analytics Module.

Provides comprehensive analytics for inventory performance monitoring,
ABC analysis, and cost optimization insights.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ABCCategory(Enum):
    """ABC Classification categories."""
    A = "A"  # High value (top 80% cumulative value)
    B = "B"  # Medium value (next 15%)
    C = "C"  # Low value (remaining 5%)


@dataclass
class SKUMetrics:
    """Comprehensive metrics for a single SKU."""
    sku_id: str
    annual_revenue: float
    annual_cost: float
    turnover_ratio: float
    days_of_supply: float
    stockout_frequency: int
    abc_category: ABCCategory
    carrying_cost: float
    ordering_cost: float


@dataclass
class InventoryHealthScore:
    """Overall inventory health assessment."""
    overall_score: float  # 0-100
    turnover_score: float
    stockout_score: float
    carrying_cost_score: float
    fill_rate_score: float
    recommendations: List[str]


class InventoryAnalytics:
    """
    Analytics engine for inventory performance analysis.
    
    Provides ABC classification, KPI calculations, and
    optimization recommendations.
    """
    
    # Industry benchmark thresholds
    TURNOVER_EXCELLENT = 12.0
    TURNOVER_GOOD = 6.0
    TURNOVER_POOR = 3.0
    
    STOCKOUT_EXCELLENT = 0.02  # 2%
    STOCKOUT_ACCEPTABLE = 0.05  # 5%
    
    def __init__(self, sku_data: List[Dict]):
        """
        Initialize analytics engine with SKU data.
        
        Args:
            sku_data: List of SKU dictionaries with demand and cost info
        """
        self.sku_data = sku_data
        self._abc_classification: Optional[Dict[str, ABCCategory]] = None
    
    def perform_abc_analysis(
        self, 
        value_field: str = 'annual_revenue'
    ) -> Dict[str, ABCCategory]:
        """
        Perform ABC classification on inventory.
        
        Args:
            value_field: Field to use for value calculation
            
        Returns:
            Dictionary mapping SKU IDs to ABC categories
        """
        # Sort by value descending
        sorted_skus = sorted(
            self.sku_data, 
            key=lambda x: x.get(value_field, 0), 
            reverse=True
        )
        
        total_value = sum(s.get(value_field, 0) for s in sorted_skus)
        if total_value == 0:
            return {s['sku_id']: ABCCategory.C for s in sorted_skus}
        
        classification = {}
        cumulative = 0
        
        for sku in sorted_skus:
            cumulative += sku.get(value_field, 0)
            cumulative_pct = cumulative / total_value
            
            if cumulative_pct <= 0.80:
                category = ABCCategory.A
            elif cumulative_pct <= 0.95:
                category = ABCCategory.B
            else:
                category = ABCCategory.C
            
            classification[sku['sku_id']] = category
        
        self._abc_classification = classification
        return classification
    
    def calculate_turnover_ratio(
        self, 
        cost_of_goods_sold: float, 
        average_inventory: float
    ) -> float:
        """
        Calculate inventory turnover ratio.
        
        Args:
            cost_of_goods_sold: Total COGS for the period
            average_inventory: Average inventory value
            
        Returns:
            Turnover ratio (higher is better)
        """
        if average_inventory <= 0:
            return 0.0
        return cost_of_goods_sold / average_inventory
    
    def calculate_days_of_supply(
        self, 
        current_inventory: float, 
        average_daily_demand: float
    ) -> float:
        """
        Calculate days of supply remaining.
        
        Args:
            current_inventory: Current inventory units
            average_daily_demand: Average daily demand units
            
        Returns:
            Number of days current inventory will last
        """
        if average_daily_demand <= 0:
            return float('inf')
        return current_inventory / average_daily_demand
    
    def calculate_fill_rate(
        self, 
        orders_fulfilled: int, 
        total_orders: int
    ) -> float:
        """
        Calculate order fill rate.
        
        Args:
            orders_fulfilled: Number of orders completely fulfilled
            total_orders: Total number of orders
            
        Returns:
            Fill rate as percentage (0-100)
        """
        if total_orders <= 0:
            return 100.0
        return (orders_fulfilled / total_orders) * 100
    
    def calculate_carrying_cost(
        self,
        average_inventory_value: float,
        carrying_cost_rate: float = 0.25
    ) -> float:
        """
        Calculate annual carrying cost.
        
        Args:
            average_inventory_value: Average inventory $ value
            carrying_cost_rate: Annual carrying cost as % of value
            
        Returns:
            Annual carrying cost in dollars
        """
        return average_inventory_value * carrying_cost_rate
    
    def calculate_stockout_cost(
        self,
        stockout_events: int,
        avg_order_value: float,
        lost_margin_rate: float = 0.30
    ) -> float:
        """
        Estimate cost of stockouts.
        
        Args:
            stockout_events: Number of stockout events
            avg_order_value: Average order value
            lost_margin_rate: Margin lost per stockout
            
        Returns:
            Estimated stockout cost
        """
        return stockout_events * avg_order_value * lost_margin_rate
    
    def get_inventory_health_score(
        self,
        metrics: Dict
    ) -> InventoryHealthScore:
        """
        Calculate overall inventory health score.
        
        Args:
            metrics: Dictionary with inventory performance metrics
            
        Returns:
            InventoryHealthScore with detailed breakdown
        """
        recommendations = []
        
        # Turnover score (0-25 points)
        turnover = metrics.get('turnover_ratio', 0)
        if turnover >= self.TURNOVER_EXCELLENT:
            turnover_score = 25
        elif turnover >= self.TURNOVER_GOOD:
            turnover_score = 20
        elif turnover >= self.TURNOVER_POOR:
            turnover_score = 10
            recommendations.append("Consider reducing slow-moving inventory")
        else:
            turnover_score = 5
            recommendations.append("Critical: Inventory turnover is very low")
        
        # Stockout score (0-25 points)
        stockout_rate = metrics.get('stockout_rate', 0)
        if stockout_rate <= self.STOCKOUT_EXCELLENT:
            stockout_score = 25
        elif stockout_rate <= self.STOCKOUT_ACCEPTABLE:
            stockout_score = 15
            recommendations.append("Review safety stock levels for key SKUs")
        else:
            stockout_score = 5
            recommendations.append("Critical: High stockout rate affecting sales")
        
        # Carrying cost score (0-25 points)
        carrying_cost_ratio = metrics.get('carrying_cost_ratio', 0)
        if carrying_cost_ratio <= 0.20:
            carrying_cost_score = 25
        elif carrying_cost_ratio <= 0.30:
            carrying_cost_score = 15
        else:
            carrying_cost_score = 5
            recommendations.append("Reduce excess inventory to lower carrying costs")
        
        # Fill rate score (0-25 points)
        fill_rate = metrics.get('fill_rate', 100)
        if fill_rate >= 98:
            fill_rate_score = 25
        elif fill_rate >= 95:
            fill_rate_score = 20
        elif fill_rate >= 90:
            fill_rate_score = 10
            recommendations.append("Improve demand forecasting accuracy")
        else:
            fill_rate_score = 5
            recommendations.append("Critical: Fill rate is impacting customer satisfaction")
        
        overall_score = turnover_score + stockout_score + carrying_cost_score + fill_rate_score
        
        return InventoryHealthScore(
            overall_score=overall_score,
            turnover_score=turnover_score,
            stockout_score=stockout_score,
            carrying_cost_score=carrying_cost_score,
            fill_rate_score=fill_rate_score,
            recommendations=recommendations
        )
    
    def get_optimization_opportunities(self) -> List[Dict]:
        """
        Identify optimization opportunities across inventory.
        
        Returns:
            List of optimization recommendations with estimated savings
        """
        opportunities = []
        
        if not self._abc_classification:
            self.perform_abc_analysis()
        
        # Identify category-specific optimizations
        a_items = [s for s in self.sku_data 
                   if self._abc_classification.get(s['sku_id']) == ABCCategory.A]
        c_items = [s for s in self.sku_data 
                   if self._abc_classification.get(s['sku_id']) == ABCCategory.C]
        
        # A items: Tighter control, more frequent review
        if a_items:
            opportunities.append({
                'type': 'Process Improvement',
                'category': 'A Items',
                'recommendation': 'Implement weekly review cycle for A items',
                'impact': 'High',
                'sku_count': len(a_items)
            })
        
        # C items: Consider consolidation or elimination
        if c_items:
            avg_c_value = np.mean([c.get('annual_revenue', 0) for c in c_items])
            opportunities.append({
                'type': 'SKU Rationalization',
                'category': 'C Items',
                'recommendation': f'Review {len(c_items)} C-items for possible elimination',
                'impact': 'Medium',
                'sku_count': len(c_items),
                'avg_value': avg_c_value
            })
        
        return opportunities
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive inventory analysis report.
        
        Returns:
            Dictionary with full analysis results
        """
        abc = self.perform_abc_analysis()
        
        # Count by category
        category_counts = {cat: 0 for cat in ABCCategory}
        for cat in abc.values():
            category_counts[cat] += 1
        
        return {
            'summary': {
                'total_skus': len(self.sku_data),
                'a_items': category_counts[ABCCategory.A],
                'b_items': category_counts[ABCCategory.B],
                'c_items': category_counts[ABCCategory.C],
            },
            'abc_classification': {k: v.value for k, v in abc.items()},
            'opportunities': self.get_optimization_opportunities()
        }
