import math

class InventoryEngine:
    def __init__(self, service_level_z: float = 1.65):
        """
        Z = 1.65 corresponds to a 95% service level.
        Z = 2.33 corresponds to a 99% service level.
        """
        self.z = service_level_z

    def calculate_base_safety_stock(
        self, 
        avg_demand: float, 
        std_demand: float, 
        avg_lead_time_days: float, 
        std_lead_time_days: float
    ) -> float:
        """
        Calculates standard safety stock accounting for both 
        demand variance and lead-time variance.
        """
        variance_component = (avg_lead_time_days * (std_demand ** 2)) + \
                             ((avg_demand ** 2) * (std_lead_time_days ** 2))
        return self.z * math.sqrt(variance_component)

    def calculate_dynamic_safety_stock(
        self, 
        base_ss: float, 
        scrap_rate: float
    ) -> float:
        """
        Scales safety stock upward based on conveyor line defect rate.
        Clamped to prevent division by zero or unrealistic blow-up.
        """
        safe_scrap = min(max(scrap_rate, 0.0), 0.50)  # cap at 50% scrap
        dynamic_ss = base_ss / (1.0 - safe_scrap)
        return round(dynamic_ss, 2)

    def calculate_reorder_point(
        self, 
        avg_demand: float, 
        avg_lead_time_days: float, 
        safety_stock: float
    ) -> float:
        """
        ROP = (Lead Time Demand) + Safety Stock
        """
        lead_time_demand = avg_demand * avg_lead_time_days
        return round(lead_time_demand + safety_stock, 2)