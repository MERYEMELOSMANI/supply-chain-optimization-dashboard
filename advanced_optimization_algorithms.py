"""
Advanced Optimization Algorithms for Supply Chain Management
This module contains various optimization algorithms for supply chain optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, linprog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pulp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SupplyChainOptimizer:
    """
    Advanced Supply Chain Optimization Class with multiple algorithms
    """
    
    def __init__(self, data):
        self.data = data
        self.agg_data = data.groupby("Product type").agg({
            "Production volumes": "sum",
            "Manufacturing costs": "mean",
            "Costs": "mean",
            "Price": "mean",
            "Revenue generated": "sum"
        }).reset_index()
        
    def linear_programming_optimization(self, demand_factor=0.8, capacities=[35000, 30000]):
        """
        Classic Linear Programming Optimization using PuLP
        """
        prob = pulp.LpProblem("LP_Supply_Chain", pulp.LpMinimize)
        
        products = self.agg_data["Product type"].tolist()
        warehouses = ['Mumbai', 'Kolkata']
        
        # Decision variables
        x = pulp.LpVariable.dicts("Production", 
                                 ((w, p) for w in warehouses for p in products), 
                                 lowBound=0, cat="Continuous")
        
        # Objective function: minimize total cost
        total_cost = sum(
            self.agg_data.loc[self.agg_data["Product type"] == p, "Manufacturing costs"].iloc[0] * x[(w, p)] +
            self.agg_data.loc[self.agg_data["Product type"] == p, "Costs"].iloc[0] * x[(w, p)]
            for w in warehouses for p in products
        )
        prob += total_cost
        
        # Constraints
        for i, w in enumerate(warehouses):
            prob += sum(x[(w, p)] for p in products) <= capacities[i], f"Capacity_{w}"
        
        for p in products:
            demand = self.agg_data.loc[self.agg_data["Product type"] == p, "Production volumes"].iloc[0] * demand_factor
            prob += sum(x[(w, p)] for w in warehouses) >= demand * 0.9, f"Demand_{p}"
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return {
            "status": pulp.LpStatus[prob.status],
            "objective": pulp.value(prob.objective) if prob.status == 1 else None,
            "variables": {str(v): v.varValue for v in prob.variables() if v.varValue and v.varValue > 0}
        }
    
    def genetic_algorithm_optimization(self, demand_factor=0.8, capacities=[35000, 30000], 
                                     generations=100, population_size=50):
        """
        Genetic Algorithm for Supply Chain Optimization using Differential Evolution
        """
        products = self.agg_data["Product type"].tolist()
        n_products = len(products)
        n_warehouses = 2
        
        def objective_function(x):
            # Reshape x to warehouse-product matrix
            allocation = x.reshape(n_warehouses, n_products)
            
            total_cost = 0
            penalty = 0
            
            # Calculate costs
            for i, warehouse in enumerate(['Mumbai', 'Kolkata']):
                for j, product in enumerate(products):
                    prod_vol = allocation[i, j]
                    
                    # Manufacturing cost
                    mfg_cost = self.agg_data.loc[self.agg_data["Product type"] == product, "Manufacturing costs"].iloc[0]
                    transport_cost = self.agg_data.loc[self.agg_data["Product type"] == product, "Costs"].iloc[0]
                    
                    total_cost += prod_vol * (mfg_cost + transport_cost)
            
            # Capacity constraints (penalties)
            for i in range(n_warehouses):
                warehouse_load = np.sum(allocation[i, :])
                if warehouse_load > capacities[i]:
                    penalty += 1000 * (warehouse_load - capacities[i]) ** 2
            
            # Demand constraints (penalties)
            for j, product in enumerate(products):
                demand = self.agg_data.loc[self.agg_data["Product type"] == product, "Production volumes"].iloc[0] * demand_factor
                total_production = np.sum(allocation[:, j])
                if total_production < demand * 0.9:
                    penalty += 1000 * (demand * 0.9 - total_production) ** 2
            
            return total_cost + penalty
        
        # Bounds for each variable (production allocation)
        max_prod = self.agg_data["Production volumes"].max() * 2
        bounds = [(0, max_prod) for _ in range(n_warehouses * n_products)]
        
        # Run genetic algorithm
        result = differential_evolution(objective_function, bounds, 
                                      maxiter=generations, popsize=population_size, 
                                      seed=42, disp=False)
        
        allocation = result.x.reshape(n_warehouses, n_products)
        
        return {
            "status": "Optimal" if result.success else "Failed",
            "objective": result.fun,
            "allocation": allocation,
            "products": products,
            "warehouses": ['Mumbai', 'Kolkata']
        }
    
    def clustering_based_optimization(self, n_clusters=3):
        """
        Clustering-based optimization to group similar products
        """
        # Prepare data for clustering
        features = ['Production volumes', 'Manufacturing costs', 'Costs', 'Price']
        X = self.agg_data[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster information to data
        self.agg_data['Cluster'] = clusters
        
        # Optimize within each cluster
        cluster_results = {}
        
        for cluster_id in range(n_clusters):
            cluster_data = self.agg_data[self.agg_data['Cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                cluster_results[f'Cluster_{cluster_id}'] = {
                    'products': cluster_data['Product type'].tolist(),
                    'avg_manufacturing_cost': cluster_data['Manufacturing costs'].mean(),
                    'avg_transport_cost': cluster_data['Costs'].mean(),
                    'total_volume': cluster_data['Production volumes'].sum(),
                    'avg_price': cluster_data['Price'].mean()
                }
        
        return {
            "clusters": cluster_results,
            "cluster_centers": kmeans.cluster_centers_,
            "scaler": scaler,
            "feature_names": features
        }
    
    def multi_objective_optimization(self, demand_factor=0.8, capacities=[35000, 30000], 
                                   cost_weight=0.6, revenue_weight=0.3, utilization_weight=0.1):
        """
        Multi-objective optimization balancing cost, revenue, and utilization
        """
        prob = pulp.LpProblem("Multi_Objective_Supply_Chain", pulp.LpMinimize)
        
        products = self.agg_data["Product type"].tolist()
        warehouses = ['Mumbai', 'Kolkata']
        
        # Decision variables
        x = pulp.LpVariable.dicts("Production", 
                                 ((w, p) for w in warehouses for p in products), 
                                 lowBound=0, cat="Continuous")
        
        # Binary variables for warehouse utilization
        y = pulp.LpVariable.dicts("Warehouse", warehouses, cat="Binary")
        
        # Objective components
        total_cost = sum(
            self.agg_data.loc[self.agg_data["Product type"] == p, "Manufacturing costs"].iloc[0] * x[(w, p)] +
            self.agg_data.loc[self.agg_data["Product type"] == p, "Costs"].iloc[0] * x[(w, p)]
            for w in warehouses for p in products
        )
        
        total_revenue = sum(
            self.agg_data.loc[self.agg_data["Product type"] == p, "Price"].iloc[0] * x[(w, p)]
            for w in warehouses for p in products
        )
        
        # Fixed costs for warehouse operation
        fixed_costs = sum(y[w] * 10000 for w in warehouses)
        
        # Multi-objective function
        prob += cost_weight * (total_cost + fixed_costs) - revenue_weight * total_revenue
        
        # Constraints
        for i, w in enumerate(warehouses):
            # Capacity constraints
            prob += sum(x[(w, p)] for p in products) <= capacities[i] * y[w], f"Capacity_{w}"
            
            # Minimum utilization if warehouse is open
            prob += sum(x[(w, p)] for p in products) >= capacities[i] * 0.3 * y[w], f"MinUtil_{w}"
        
        for p in products:
            demand = self.agg_data.loc[self.agg_data["Product type"] == p, "Production volumes"].iloc[0] * demand_factor
            prob += sum(x[(w, p)] for w in warehouses) >= demand * 0.9, f"Demand_{p}"
        
        # At least one warehouse must be open
        prob += sum(y[w] for w in warehouses) >= 1, "MinWarehouses"
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return {
            "status": pulp.LpStatus[prob.status],
            "objective": pulp.value(prob.objective) if prob.status == 1 else None,
            "production_variables": {str(v): v.varValue for v in prob.variables() if "Production" in str(v) and v.varValue and v.varValue > 0},
            "warehouse_variables": {str(v): v.varValue for v in prob.variables() if "Warehouse" in str(v)}
        }
    
    def robust_optimization(self, demand_scenarios, capacities=[35000, 30000]):
        """
        Robust optimization considering multiple demand scenarios
        """
        prob = pulp.LpProblem("Robust_Supply_Chain", pulp.LpMinimize)
        
        products = self.agg_data["Product type"].tolist()
        warehouses = ['Mumbai', 'Kolkata']
        scenarios = list(demand_scenarios.keys())
        
        # First-stage variables (production decisions)
        x = pulp.LpVariable.dicts("Production", 
                                 ((w, p) for w in warehouses for p in products), 
                                 lowBound=0, cat="Continuous")
        
        # Second-stage variables (scenario-specific adjustments)
        z = pulp.LpVariable.dicts("Adjustment", 
                                 ((s, w, p) for s in scenarios for w in warehouses for p in products), 
                                 lowBound=0, cat="Continuous")
        
        # Objective: minimize worst-case cost
        max_cost = pulp.LpVariable("MaxCost", cat="Continuous")
        prob += max_cost
        
        # Constraints for each scenario
        for s in scenarios:
            scenario_cost = sum(
                self.agg_data.loc[self.agg_data["Product type"] == p, "Manufacturing costs"].iloc[0] * 
                (x[(w, p)] + z[(s, w, p)]) +
                self.agg_data.loc[self.agg_data["Product type"] == p, "Costs"].iloc[0] * 
                (x[(w, p)] + z[(s, w, p)])
                for w in warehouses for p in products
            )
            
            prob += max_cost >= scenario_cost, f"ScenarioCost_{s}"
            
            # Capacity constraints for each scenario
            for i, w in enumerate(warehouses):
                prob += sum(x[(w, p)] + z[(s, w, p)] for p in products) <= capacities[i], f"Capacity_{s}_{w}"
            
            # Demand constraints for each scenario
            for p in products:
                demand = self.agg_data.loc[self.agg_data["Product type"] == p, "Production volumes"].iloc[0] * demand_scenarios[s]
                prob += sum(x[(w, p)] + z[(s, w, p)] for w in warehouses) >= demand * 0.9, f"Demand_{s}_{p}"
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return {
            "status": pulp.LpStatus[prob.status],
            "objective": pulp.value(prob.objective) if prob.status == 1 else None,
            "base_production": {str(v): v.varValue for v in prob.variables() if "Production" in str(v) and v.varValue and v.varValue > 0},
            "scenario_adjustments": {str(v): v.varValue for v in prob.variables() if "Adjustment" in str(v) and v.varValue and v.varValue > 0}
        }
    
    def sensitivity_analysis(self, base_params, param_ranges):
        """
        Perform sensitivity analysis on key parameters
        """
        results = {}
        
        for param_name, param_range in param_ranges.items():
            param_results = []
            
            for param_value in param_range:
                # Update parameters
                current_params = base_params.copy()
                current_params[param_name] = param_value
                
                # Run optimization with updated parameters
                result = self.linear_programming_optimization(
                    demand_factor=current_params.get('demand_factor', 0.8),
                    capacities=current_params.get('capacities', [35000, 30000])
                )
                
                param_results.append({
                    'parameter_value': param_value,
                    'objective': result['objective'],
                    'status': result['status']
                })
            
            results[param_name] = param_results
        
        return results
    
    def monte_carlo_simulation(self, n_simulations=1000, demand_factor_range=(0.5, 1.2), 
                              cost_variation=0.1, capacities=[35000, 30000]):
        """
        Monte Carlo simulation for uncertainty analysis
        """
        results = []
        
        for i in range(n_simulations):
            # Random demand factor
            demand_factor = np.random.uniform(*demand_factor_range)
            
            # Random cost variations
            cost_multipliers = np.random.normal(1.0, cost_variation, len(self.agg_data))
            
            # Create temporary data with cost variations
            temp_data = self.agg_data.copy()
            temp_data['Manufacturing costs'] *= cost_multipliers
            temp_data['Costs'] *= cost_multipliers
            
            # Create temporary optimizer
            temp_optimizer = SupplyChainOptimizer(pd.DataFrame())
            temp_optimizer.agg_data = temp_data
            
            # Run optimization
            result = temp_optimizer.linear_programming_optimization(demand_factor, capacities)
            
            if result['status'] == 'Optimal':
                results.append({
                    'simulation': i,
                    'demand_factor': demand_factor,
                    'objective': result['objective'],
                    'cost_multipliers': cost_multipliers.tolist()
                })
        
        return {
            'simulations': results,
            'statistics': {
                'mean_objective': np.mean([r['objective'] for r in results]),
                'std_objective': np.std([r['objective'] for r in results]),
                'min_objective': np.min([r['objective'] for r in results]),
                'max_objective': np.max([r['objective'] for r in results]),
                'success_rate': len(results) / n_simulations
            }
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data (you would replace this with your actual data loading)
    try:
        data = pd.read_csv("supply_chain_data.csv")
        optimizer = SupplyChainOptimizer(data)
        
        print("=== Supply Chain Optimization Results ===\n")
        
        # 1. Linear Programming
        print("1. Linear Programming Optimization:")
        lp_result = optimizer.linear_programming_optimization()
        print(f"Status: {lp_result['status']}")
        print(f"Objective: ${lp_result['objective']:,.2f}\n" if lp_result['objective'] else "No solution\n")
        
        # 2. Genetic Algorithm
        print("2. Genetic Algorithm Optimization:")
        ga_result = optimizer.genetic_algorithm_optimization()
        print(f"Status: {ga_result['status']}")
        print(f"Objective: ${ga_result['objective']:,.2f}\n" if ga_result['objective'] else "No solution\n")
        
        # 3. Clustering Analysis
        print("3. Clustering-based Analysis:")
        cluster_result = optimizer.clustering_based_optimization()
        print(f"Number of clusters: {len(cluster_result['clusters'])}")
        for cluster_name, cluster_info in cluster_result['clusters'].items():
            print(f"{cluster_name}: {len(cluster_info['products'])} products")
        print()
        
        # 4. Multi-objective Optimization
        print("4. Multi-objective Optimization:")
        mo_result = optimizer.multi_objective_optimization()
        print(f"Status: {mo_result['status']}")
        print(f"Objective: ${mo_result['objective']:,.2f}\n" if mo_result['objective'] else "No solution\n")
        
        # 5. Robust Optimization
        print("5. Robust Optimization:")
        demand_scenarios = {
            'low': 0.6,
            'medium': 0.8,
            'high': 1.0,
            'peak': 1.2
        }
        robust_result = optimizer.robust_optimization(demand_scenarios)
        print(f"Status: {robust_result['status']}")
        print(f"Worst-case Objective: ${robust_result['objective']:,.2f}\n" if robust_result['objective'] else "No solution\n")
        
        # 6. Monte Carlo Simulation
        print("6. Monte Carlo Simulation (100 runs):")
        mc_result = optimizer.monte_carlo_simulation(n_simulations=100)
        stats = mc_result['statistics']
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Mean Objective: ${stats['mean_objective']:,.2f}")
        print(f"Std Deviation: ${stats['std_objective']:,.2f}")
        print(f"Range: ${stats['min_objective']:,.2f} - ${stats['max_objective']:,.2f}")
        
    except FileNotFoundError:
        print("Error: supply_chain_data.csv not found. Please ensure the data file is in the same directory.")
    except Exception as e:
        print(f"Error: {str(e)}")
