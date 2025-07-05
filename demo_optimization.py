"""
Supply Chain Optimization Demo
Run this script to see different optimization algorithms in action
"""

import pandas as pd
import numpy as np
from advanced_optimization_algorithms import SupplyChainOptimizer
import matplotlib.pyplot as plt

def main():
    print("🚀 Supply Chain Optimization Demo")
    print("=" * 50)
    
    # Load data
    try:
        data = pd.read_csv("supply_chain_data.csv")
        optimizer = SupplyChainOptimizer(data)
        
        print(f"📊 Data loaded successfully!")
        print(f"   - Total records: {len(data)}")
        print(f"   - Product types: {data['Product type'].nunique()}")
        print(f"   - Locations: {data['Location'].nunique()}")
        print()
        
        # Basic parameters
        demand_factor = 0.8
        capacities = [35000, 30000]  # Mumbai, Kolkata
        
        print("🔧 Running optimization algorithms...")
        print("-" * 30)
        
        # 1. Linear Programming
        print("1️⃣ Linear Programming Optimization")
        lp_result = optimizer.linear_programming_optimization(demand_factor, capacities)
        print(f"   Status: {lp_result['status']}")
        if lp_result['objective']:
            print(f"   Total Cost: ${lp_result['objective']:,.2f}")
            print(f"   Active Variables: {len(lp_result['variables'])}")
        print()
        
        # 2. Genetic Algorithm
        print("2️⃣ Genetic Algorithm Optimization")
        ga_result = optimizer.genetic_algorithm_optimization(demand_factor, capacities, 
                                                           generations=50, population_size=30)
        print(f"   Status: {ga_result['status']}")
        if ga_result['objective']:
            print(f"   Total Cost: ${ga_result['objective']:,.2f}")
            print(f"   Allocation Matrix Shape: {ga_result['allocation'].shape}")
        print()
        
        # 3. Clustering Analysis
        print("3️⃣ Clustering-based Analysis")
        cluster_result = optimizer.clustering_based_optimization(n_clusters=3)
        print(f"   Number of clusters: {len(cluster_result['clusters'])}")
        for cluster_name, cluster_info in cluster_result['clusters'].items():
            print(f"   {cluster_name}: {len(cluster_info['products'])} products, "
                  f"Avg Cost: ${cluster_info['avg_manufacturing_cost']:.2f}")
        print()
        
        # 4. Multi-objective Optimization
        print("4️⃣ Multi-objective Optimization")
        mo_result = optimizer.multi_objective_optimization(demand_factor, capacities)
        print(f"   Status: {mo_result['status']}")
        if mo_result['objective']:
            print(f"   Weighted Objective: ${mo_result['objective']:,.2f}")
            active_warehouses = sum(1 for v in mo_result['warehouse_variables'].values() if v > 0)
            print(f"   Active Warehouses: {active_warehouses}")
        print()
        
        # 5. Robust Optimization
        print("5️⃣ Robust Optimization")
        demand_scenarios = {
            'conservative': 0.6,
            'baseline': 0.8,
            'optimistic': 1.0,
            'peak': 1.2
        }
        robust_result = optimizer.robust_optimization(demand_scenarios, capacities)
        print(f"   Status: {robust_result['status']}")
        if robust_result['objective']:
            print(f"   Worst-case Cost: ${robust_result['objective']:,.2f}")
            print(f"   Scenarios considered: {len(demand_scenarios)}")
        print()
        
        # 6. Sensitivity Analysis
        print("6️⃣ Sensitivity Analysis")
        base_params = {
            'demand_factor': 0.8,
            'capacities': [35000, 30000]
        }
        param_ranges = {
            'demand_factor': np.linspace(0.5, 1.2, 8)
        }
        
        sensitivity_result = optimizer.sensitivity_analysis(base_params, param_ranges)
        
        for param_name, results in sensitivity_result.items():
            optimal_count = sum(1 for r in results if r['status'] == 'Optimal')
            print(f"   {param_name}: {optimal_count}/{len(results)} optimal solutions")
            if optimal_count > 0:
                objectives = [r['objective'] for r in results if r['status'] == 'Optimal']
                print(f"   Cost range: ${min(objectives):,.2f} - ${max(objectives):,.2f}")
        print()
        
        # 7. Monte Carlo Simulation
        print("7️⃣ Monte Carlo Simulation")
        mc_result = optimizer.monte_carlo_simulation(n_simulations=100)
        stats = mc_result['statistics']
        print(f"   Simulations: 100")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Mean Cost: ${stats['mean_objective']:,.2f}")
        print(f"   Std Deviation: ${stats['std_objective']:,.2f}")
        print(f"   Cost Range: ${stats['min_objective']:,.2f} - ${stats['max_objective']:,.2f}")
        print()
        
        # Results Summary
        print("📊 Results Summary")
        print("=" * 50)
        
        results_summary = []
        if lp_result['objective']:
            results_summary.append(("Linear Programming", lp_result['objective']))
        if ga_result['objective']:
            results_summary.append(("Genetic Algorithm", ga_result['objective']))
        if mo_result['objective']:
            results_summary.append(("Multi-objective", mo_result['objective']))
        if robust_result['objective']:
            results_summary.append(("Robust Optimization", robust_result['objective']))
        
        if results_summary:
            print("Cost Comparison:")
            for method, cost in sorted(results_summary, key=lambda x: x[1]):
                print(f"   {method:<20}: ${cost:,.2f}")
            
            best_method = min(results_summary, key=lambda x: x[1])
            print(f"\n🏆 Best Method: {best_method[0]} (${best_method[1]:,.2f})")
        
        print("\n✅ Optimization demo completed successfully!")
        print("\nNext steps:")
        print("   - Run 'streamlit run optimization_dashboard.py' for interactive dashboard")
        print("   - Modify parameters in the algorithms for different scenarios")
        print("   - Analyze the clustering results for product grouping insights")
        
    except FileNotFoundError:
        print("❌ Error: supply_chain_data.csv not found!")
        print("Please ensure the data file is in the same directory.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your data file and dependencies.")

if __name__ == "__main__":
    main()
