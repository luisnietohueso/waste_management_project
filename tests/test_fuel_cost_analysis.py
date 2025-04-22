"""
Author: Luis Nieto Hueso  
Date: 28/03/2025  
Description:  
This test script validates the functionality of the fuel cost analysis utilities.  
It confirms that the fuel savings calculation logic behaves correctly and that visual and markdown summaries  
are successfully generated.

Function: `test_analyze_fuel_cost_reduction()`  
- Simulates a scenario where route optimisation reduces total travel distance from 15.0 km to 10.0 km.  
- Calculates fuel consumption and cost before and after optimisation using the `analyze_fuel_cost_reduction` function.  
- Packs the results into a metrics dictionary to generate a markdown summary (`fuel_summary.md`) and a plot (`fuel_cost_plot.png`).  
- Uses assertions to verify that:  
  - The reduction percentage is a valid float between 0 and 100  
  - The cost after optimisation is lower than before, confirming savings

Utilities Used:  
- `analyze_fuel_cost_reduction` for computing fuel metrics  
- `generate_fuel_summary_report` and `generate_fuel_summary_plot` for producing report outputs

This test helps ensure that the cost-saving logic produces realistic, valid results and supports  
the generation of clear, visual documentation for reporting purposes.
"""


from utils.helpers import analyze_fuel_cost_reduction
from utils.fuel_summary import generate_fuel_summary_report, generate_fuel_summary_plot

def test_analyze_fuel_cost_reduction():
    optimized_distance = 10.0
    static_distance = 15.0
    fuel_efficiency = 0.35
    fuel_price = 1.50

    fuel_before, fuel_after, cost_before, cost_after, reduction = (
        analyze_fuel_cost_reduction(optimized_distance, static_distance, fuel_efficiency, fuel_price)
    )

    metrics = {
        "Fuel Consumption Before": fuel_before,
        "Fuel Consumption After": fuel_after,
        "Fuel Cost Before": cost_before,
        "Fuel Cost After": cost_after,
        "Reduction (%)": reduction
    }

    generate_fuel_summary_report(metrics)
    generate_fuel_summary_plot(metrics)

    assert isinstance(reduction, float)
    assert 0 <= reduction <= 100
    assert cost_after < cost_before
