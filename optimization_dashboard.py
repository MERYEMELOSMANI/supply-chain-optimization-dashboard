import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Supply Chain Optimization", layout="wide")

st.title("🚀 Advanced Supply Chain Optimization Dashboard")
st.write("Optimize production and transportation for haircare, skincare, cosmetics across Mumbai and Kolkata.")

# Load and process dataset
@st.cache_data
def load_data():
    data = pd.read_csv("supply_chain_data.csv")
    return data

data = load_data()

# Data overview
with st.expander("📊 Data Overview"):
    st.write(f"Total Records: {len(data)}")
    st.write(f"Product Types: {data['Product type'].nunique()}")
    st.write(f"Locations: {data['Location'].nunique()}")
    st.dataframe(data.head())

# Process data for optimization
agg_data = data.groupby("Product type").agg({
    "Production volumes": "sum",
    "Manufacturing costs": "mean",
    "Costs": "mean",
    "Price": "mean",
    "Revenue generated": "sum"
}).reset_index()
total_supply = agg_data["Production volumes"].sum()

# Enhanced Inputs Section
st.sidebar.header("🎛️ Optimization Parameters")

# Market demand settings
st.sidebar.subheader("Market Demand")
demand_factor = st.sidebar.slider("Overall Demand Factor", 0.1, 2.0, 0.8, 0.1, 
                                 help="Multiplier for total market demand")
seasonal_adjustment = st.sidebar.slider("Seasonal Adjustment", 0.5, 1.5, 1.0, 0.1,
                                       help="Seasonal demand variation")

# Capacity settings
st.sidebar.subheader("Warehouse Capacities")
capacity_w1 = st.sidebar.slider("Mumbai Capacity", 10000, 100000, 35000, 5000)
capacity_w2 = st.sidebar.slider("Kolkata Capacity", 10000, 100000, 30000, 5000)

# Cost settings
st.sidebar.subheader("Cost Parameters")
transport_cost_multiplier = st.sidebar.slider("Transport Cost Factor", 0.5, 2.0, 1.0, 0.1)
inventory_cost_rate = st.sidebar.slider("Inventory Cost Rate (%)", 1, 20, 5, 1) / 100

# Optimization objectives
st.sidebar.subheader("Optimization Objectives")
obj_weights = {
    "cost": st.sidebar.slider("Cost Minimization Weight", 0.0, 1.0, 0.6, 0.1),
    "revenue": st.sidebar.slider("Revenue Maximization Weight", 0.0, 1.0, 0.3, 0.1),
    "utilization": st.sidebar.slider("Capacity Utilization Weight", 0.0, 1.0, 0.1, 0.1)
}

# Product-specific demand
st.sidebar.subheader("Product Demand Distribution")
product_demand = {}
for product in agg_data["Product type"].unique():
    product_demand[product] = st.sidebar.slider(
        f"{product.title()} Demand Factor", 
        0.1, 2.0, 1.0, 0.1,
        key=f"demand_{product}"
    )

# Advanced Optimization Model
def create_optimization_model():
    # Create the optimization problem
    prob = pulp.LpProblem("Advanced_Supply_Chain_Optimization", pulp.LpMinimize)
    
    # Decision variables
    products = agg_data["Product type"].tolist()
    warehouses = ['Mumbai', 'Kolkata']
    
    # Production allocation variables
    production_vars = pulp.LpVariable.dicts("Production", 
                                           ((w, p) for w in warehouses for p in products), 
                                           lowBound=0, cat="Continuous")
    
    # Inventory variables
    inventory_vars = pulp.LpVariable.dicts("Inventory", 
                                         ((w, p) for w in warehouses for p in products), 
                                         lowBound=0, cat="Continuous")
    
    # Binary variables for warehouse operation
    warehouse_operation = pulp.LpVariable.dicts("WarehouseOp", warehouses, cat="Binary")
    
    # Multi-objective function
    manufacturing_cost = sum(
        agg_data.loc[agg_data["Product type"] == p, "Manufacturing costs"].iloc[0] * 
        production_vars[(w, p)] for w in warehouses for p in products
    )
    
    transportation_cost = sum(
        agg_data.loc[agg_data["Product type"] == p, "Costs"].iloc[0] * 
        transport_cost_multiplier * production_vars[(w, p)] 
        for w in warehouses for p in products
    )
    
    inventory_cost = sum(
        inventory_vars[(w, p)] * inventory_cost_rate * 
        agg_data.loc[agg_data["Product type"] == p, "Price"].iloc[0]
        for w in warehouses for p in products
    )
    
    revenue = sum(
        production_vars[(w, p)] * 
        agg_data.loc[agg_data["Product type"] == p, "Price"].iloc[0]
        for w in warehouses for p in products
    )
    
    # Fixed costs for warehouse operation
    fixed_cost = sum(warehouse_operation[w] * 10000 for w in warehouses)
    
    # Weighted objective function
    total_cost = (obj_weights["cost"] * (manufacturing_cost + transportation_cost + inventory_cost + fixed_cost) -
                  obj_weights["revenue"] * revenue)
    
    prob += total_cost, "Multi_Objective_Function"
    
    # Constraints
    capacities = {'Mumbai': capacity_w1, 'Kolkata': capacity_w2}
    
    # Capacity constraints
    for w in warehouses:
        prob += (sum(production_vars[(w, p)] for p in products) <= 
                capacities[w] * warehouse_operation[w]), f"Capacity_{w}"
    
    # Demand constraints with product-specific factors
    total_demand = total_supply * demand_factor * seasonal_adjustment
    for p in products:
        product_specific_demand = (
            agg_data.loc[agg_data["Product type"] == p, "Production volumes"].iloc[0] * 
            demand_factor * product_demand[p] * seasonal_adjustment
        )
        prob += (sum(production_vars[(w, p)] for w in warehouses) >= 
                product_specific_demand * 0.8), f"Min_Demand_{p}"
        prob += (sum(production_vars[(w, p)] for w in warehouses) <= 
                product_specific_demand * 1.2), f"Max_Demand_{p}"
    
    # Supply constraints
    for p in products:
        max_supply = agg_data.loc[agg_data["Product type"] == p, "Production volumes"].iloc[0] * 1.5
        prob += (sum(production_vars[(w, p)] for w in warehouses) <= max_supply), f"Supply_{p}"
    
    # Inventory balance constraints
    for w in warehouses:
        for p in products:
            prob += (inventory_vars[(w, p)] >= 
                    production_vars[(w, p)] * 0.1), f"Min_Inventory_{w}_{p}"
            prob += (inventory_vars[(w, p)] <= 
                    production_vars[(w, p)] * 0.3), f"Max_Inventory_{w}_{p}"
    
    # At least one warehouse must operate
    prob += sum(warehouse_operation[w] for w in warehouses) >= 1, "Min_Warehouses"
    
    return prob, production_vars, inventory_vars, warehouse_operation, products, warehouses

# Create and solve the optimization model
with st.spinner("🔄 Running optimization..."):
    prob, production_vars, inventory_vars, warehouse_operation, products, warehouses = create_optimization_model()
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Enhanced Results Section
st.header("📈 Optimization Results")

if pulp.LpStatus[prob.status] == "Optimal":
    st.success("✅ Optimal solution found!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = pulp.value(prob.objective)
    total_production = sum(production_vars[(w, p)].varValue or 0 
                         for w in warehouses for p in products)
    total_revenue = sum((production_vars[(w, p)].varValue or 0) * 
                       agg_data.loc[agg_data["Product type"] == p, "Price"].iloc[0]
                       for w in warehouses for p in products)
    profit = total_revenue - total_cost
    
    with col1:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with col2:
        st.metric("Total Production", f"{total_production:,.0f}")
    with col3:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col4:
        st.metric("Profit", f"${profit:,.2f}")
    
    # Production allocation table
    st.subheader("🏭 Production Allocation")
    allocation_data = []
    for w in warehouses:
        for p in products:
            value = production_vars[(w, p)].varValue or 0
            if value > 0:
                allocation_data.append({
                    'Warehouse': w,
                    'Product': p.title(),
                    'Quantity': value,
                    'Cost per Unit': agg_data.loc[agg_data["Product type"] == p, "Manufacturing costs"].iloc[0],
                    'Total Cost': value * agg_data.loc[agg_data["Product type"] == p, "Manufacturing costs"].iloc[0]
                })
    
    if allocation_data:
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df.style.format({
            'Quantity': '{:.0f}',
            'Cost per Unit': '${:.2f}',
            'Total Cost': '${:.2f}'
        }))
        
        # Warehouse utilization
        st.subheader("🏢 Warehouse Utilization")
        fig = go.Figure()
        
        utilization_data = []
        for w in warehouses:
            total_allocated = sum(production_vars[(w, p)].varValue or 0 for p in products)
            capacity = capacity_w1 if w == 'Mumbai' else capacity_w2
            utilization = (total_allocated / capacity) * 100 if capacity > 0 else 0
            utilization_data.append(utilization)
            
            fig.add_trace(go.Bar(
                x=[w],
                y=[utilization],
                text=[f'{utilization:.1f}%'],
                textposition='auto',
                name=w,
                marker_color='lightblue' if utilization < 80 else 'orange' if utilization < 95 else 'red'
            ))
        
        fig.update_layout(
            title="Warehouse Capacity Utilization",
            yaxis_title="Utilization (%)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Product distribution pie chart
        st.subheader("📊 Product Distribution")
        product_totals = {}
        for p in products:
            total = sum(production_vars[(w, p)].varValue or 0 for w in warehouses)
            if total > 0:
                product_totals[p.title()] = total
        
        if product_totals:
            fig_pie = px.pie(
                values=list(product_totals.values()),
                names=list(product_totals.keys()),
                title="Production Distribution by Product Type"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Cost breakdown
        st.subheader("💰 Cost Breakdown")
        manufacturing_cost = sum(
            (production_vars[(w, p)].varValue or 0) * 
            agg_data.loc[agg_data["Product type"] == p, "Manufacturing costs"].iloc[0]
            for w in warehouses for p in products
        )
        
        transportation_cost = sum(
            (production_vars[(w, p)].varValue or 0) * 
            agg_data.loc[agg_data["Product type"] == p, "Costs"].iloc[0] * transport_cost_multiplier
            for w in warehouses for p in products
        )
        
        inventory_cost = sum(
            (inventory_vars[(w, p)].varValue or 0) * inventory_cost_rate * 
            agg_data.loc[agg_data["Product type"] == p, "Price"].iloc[0]
            for w in warehouses for p in products
        )
        
        fixed_cost = sum((warehouse_operation[w].varValue or 0) * 10000 for w in warehouses)
        
        cost_data = {
            'Manufacturing': manufacturing_cost,
            'Transportation': transportation_cost,
            'Inventory': inventory_cost,
            'Fixed Costs': fixed_cost
        }
        
        fig_cost = px.bar(
            x=list(cost_data.keys()),
            y=list(cost_data.values()),
            title="Cost Breakdown Analysis"
        )
        fig_cost.update_layout(yaxis_title="Cost ($)")
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Sensitivity analysis
        st.subheader("🔍 Sensitivity Analysis")
        with st.expander("View Sensitivity Analysis"):
            st.write("**Key Insights:**")
            
            # Calculate key ratios
            profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
            avg_utilization = sum(utilization_data) / len(utilization_data)
            
            st.write(f"- Profit Margin: {profit_margin:.2f}%")
            st.write(f"- Average Warehouse Utilization: {avg_utilization:.1f}%")
            st.write(f"- Cost per Unit Produced: ${total_cost / total_production:.2f}" if total_production > 0 else "- No production")
            
            if avg_utilization > 90:
                st.warning("⚠️ High warehouse utilization - consider expanding capacity")
            elif avg_utilization < 50:
                st.info("ℹ️ Low warehouse utilization - consider consolidating operations")
            
            if profit_margin < 10:
                st.warning("⚠️ Low profit margin - review cost structure")
            elif profit_margin > 30:
                st.success("✅ Excellent profit margin")

else:
    st.error("❌ No optimal solution found")
    st.write(f"Solver Status: {pulp.LpStatus[prob.status]}")
    st.write("**Possible solutions:**")
    st.write("- Increase warehouse capacities")
    st.write("- Adjust demand factors")
    st.write("- Review cost parameters")
    st.write("- Check data quality")

# Additional optimization features
st.header("🧮 Advanced Optimization Features")

# Scenario Analysis
st.subheader("📊 Scenario Analysis")
with st.expander("Run Scenario Analysis"):
    scenarios = {
        "Conservative": {"demand_mult": 0.7, "cost_mult": 1.2, "capacity_mult": 1.0},
        "Baseline": {"demand_mult": 1.0, "cost_mult": 1.0, "capacity_mult": 1.0},
        "Optimistic": {"demand_mult": 1.3, "cost_mult": 0.9, "capacity_mult": 1.0},
        "Capacity_Expansion": {"demand_mult": 1.0, "cost_mult": 1.0, "capacity_mult": 1.5}
    }
    
    if st.button("Run All Scenarios"):
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            with st.spinner(f"Running {scenario_name} scenario..."):
                # Temporarily modify parameters
                temp_demand = demand_factor * params["demand_mult"]
                temp_capacity_w1 = capacity_w1 * params["capacity_mult"]
                temp_capacity_w2 = capacity_w2 * params["capacity_mult"]
                
                # Create scenario-specific model
                scenario_prob = pulp.LpProblem(f"Scenario_{scenario_name}", pulp.LpMinimize)
                scenario_products = agg_data["Product type"].tolist()
                scenario_warehouses = ['Mumbai', 'Kolkata']
                
                scenario_vars = pulp.LpVariable.dicts(f"Scenario_{scenario_name}", 
                                                    ((w, p) for w in scenario_warehouses for p in scenario_products), 
                                                    lowBound=0, cat="Continuous")
                
                # Objective with scenario multipliers
                scenario_cost = sum(
                    agg_data.loc[agg_data["Product type"] == p, "Manufacturing costs"].iloc[0] * 
                    params["cost_mult"] * scenario_vars[(w, p)]
                    for w in scenario_warehouses for p in scenario_products
                )
                
                scenario_prob += scenario_cost, f"Scenario_{scenario_name}_Cost"
                
                # Constraints
                scenario_capacities = {'Mumbai': temp_capacity_w1, 'Kolkata': temp_capacity_w2}
                for w in scenario_warehouses:
                    scenario_prob += (sum(scenario_vars[(w, p)] for p in scenario_products) <= 
                                    scenario_capacities[w]), f"Scenario_{scenario_name}_Capacity_{w}"
                
                for p in scenario_products:
                    scenario_demand = (agg_data.loc[agg_data["Product type"] == p, "Production volumes"].iloc[0] * 
                                     temp_demand)
                    scenario_prob += (sum(scenario_vars[(w, p)] for w in scenario_warehouses) >= 
                                    scenario_demand * 0.8), f"Scenario_{scenario_name}_Demand_{p}"
                
                # Solve scenario
                scenario_prob.solve(pulp.PULP_CBC_CMD(msg=0))
                
                if pulp.LpStatus[scenario_prob.status] == "Optimal":
                    scenario_cost_val = pulp.value(scenario_prob.objective)
                    scenario_production = sum(scenario_vars[(w, p)].varValue or 0 
                                           for w in scenario_warehouses for p in scenario_products)
                    scenario_results[scenario_name] = {
                        "cost": scenario_cost_val,
                        "production": scenario_production,
                        "status": "Optimal"
                    }
                else:
                    scenario_results[scenario_name] = {
                        "cost": None,
                        "production": None,
                        "status": pulp.LpStatus[scenario_prob.status]
                    }
        
        # Display scenario comparison
        st.subheader("Scenario Comparison")
        scenario_df = pd.DataFrame(scenario_results).T
        scenario_df = scenario_df.reset_index().rename(columns={'index': 'Scenario'})
        st.dataframe(scenario_df)
        
        # Scenario visualization
        if any(result["status"] == "Optimal" for result in scenario_results.values()):
            optimal_scenarios = {k: v for k, v in scenario_results.items() if v["status"] == "Optimal"}
            
            fig_scenario = go.Figure()
            fig_scenario.add_trace(go.Bar(
                x=list(optimal_scenarios.keys()),
                y=[result["cost"] for result in optimal_scenarios.values()],
                name="Total Cost",
                yaxis="y"
            ))
            
            fig_scenario.add_trace(go.Scatter(
                x=list(optimal_scenarios.keys()),
                y=[result["production"] for result in optimal_scenarios.values()],
                mode='lines+markers',
                name="Total Production",
                yaxis="y2",
                line=dict(color='red')
            ))
            
            fig_scenario.update_layout(
                title="Scenario Analysis Results",
                yaxis=dict(title="Total Cost ($)", side="left"),
                yaxis2=dict(title="Total Production", side="right", overlaying="y"),
                height=500
            )
            st.plotly_chart(fig_scenario, use_container_width=True)

# What-if Analysis
st.subheader("🤔 What-if Analysis")
with st.expander("Interactive What-if Analysis"):
    st.write("Analyze the impact of changing key parameters:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        whatif_demand = st.slider("What-if Demand Factor", 0.1, 2.0, demand_factor, 0.1, key="whatif_demand")
        whatif_transport_cost = st.slider("What-if Transport Cost", 0.5, 2.0, transport_cost_multiplier, 0.1, key="whatif_transport")
    
    with col2:
        whatif_mumbai_capacity = st.slider("What-if Mumbai Capacity", 10000, 100000, capacity_w1, 5000, key="whatif_mumbai")
        whatif_kolkata_capacity = st.slider("What-if Kolkata Capacity", 10000, 100000, capacity_w2, 5000, key="whatif_kolkata")
    
    if st.button("Calculate What-if Results"):
        # Calculate impact
        original_total_cost = total_cost if 'total_cost' in locals() else 0
        
        # Simple impact estimation (this could be enhanced with full re-optimization)
        demand_impact = (whatif_demand - demand_factor) * total_supply * 0.1
        transport_impact = (whatif_transport_cost - transport_cost_multiplier) * original_total_cost * 0.3
        capacity_impact = ((whatif_mumbai_capacity + whatif_kolkata_capacity) - (capacity_w1 + capacity_w2)) * 0.001
        
        total_impact = demand_impact + transport_impact - capacity_impact
        new_estimated_cost = original_total_cost + total_impact
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Cost Change", f"${total_impact:,.2f}", 
                     f"{(total_impact/original_total_cost)*100:.1f}%" if original_total_cost > 0 else "N/A")
        with col2:
            st.metric("New Estimated Total Cost", f"${new_estimated_cost:,.2f}")
        with col3:
            change_percent = ((new_estimated_cost - original_total_cost) / original_total_cost * 100) if original_total_cost > 0 else 0
            st.metric("Percentage Change", f"{change_percent:.1f}%")

# Risk Analysis
st.subheader("⚠️ Risk Analysis")
with st.expander("View Risk Assessment"):
    st.write("**Identified Risks and Mitigation Strategies:**")
    
    # Calculate risk factors based on current optimization
    if 'utilization_data' in locals():
        avg_utilization = sum(utilization_data) / len(utilization_data) if utilization_data else 0
        
        risks = []
        
        if avg_utilization > 90:
            risks.append({
                "Risk": "High Capacity Utilization",
                "Impact": "High",
                "Probability": "Medium",
                "Mitigation": "Expand warehouse capacity or improve demand forecasting"
            })
        
        if transport_cost_multiplier > 1.5:
            risks.append({
                "Risk": "High Transportation Costs",
                "Impact": "Medium",
                "Probability": "High",
                "Mitigation": "Optimize routes or negotiate better rates with carriers"
            })
        
        if demand_factor < 0.5:
            risks.append({
                "Risk": "Low Demand Scenario",
                "Impact": "High",
                "Probability": "Low",
                "Mitigation": "Diversify product portfolio or expand to new markets"
            })
        
        if inventory_cost_rate > 0.15:
            risks.append({
                "Risk": "High Inventory Costs",
                "Impact": "Medium",
                "Probability": "Medium",
                "Mitigation": "Implement just-in-time inventory management"
            })
        
        if risks:
            risk_df = pd.DataFrame(risks)
            st.dataframe(risk_df)
        else:
            st.success("✅ No significant risks identified in current configuration")

# Performance Benchmarks
st.subheader("📏 Performance Benchmarks")
with st.expander("View Industry Benchmarks"):
    st.write("**Compare your optimization results with industry standards:**")
    
    if 'profit_margin' in locals() and 'avg_utilization' in locals():
        benchmarks = {
            "Metric": ["Profit Margin", "Warehouse Utilization", "Inventory Turnover", "Cost per Unit"],
            "Your Performance": [f"{profit_margin:.1f}%", f"{avg_utilization:.1f}%", "N/A", 
                               f"${total_cost / total_production:.2f}" if total_production > 0 else "N/A"],
            "Industry Average": ["15-25%", "70-85%", "8-12x", "$2.50-4.00"],
            "Best in Class": ["30%+", "85-95%", "15x+", "$1.50-2.50"],
            "Status": []
        }
        
        # Determine status for each metric
        if profit_margin >= 30:
            benchmarks["Status"].append("🏆 Best in Class")
        elif profit_margin >= 15:
            benchmarks["Status"].append("✅ Above Average")
        else:
            benchmarks["Status"].append("⚠️ Below Average")
        
        if avg_utilization >= 85:
            benchmarks["Status"].append("🏆 Best in Class")
        elif avg_utilization >= 70:
            benchmarks["Status"].append("✅ Above Average")
        else:
            benchmarks["Status"].append("⚠️ Below Average")
        
        benchmarks["Status"].extend(["📊 Data Needed", "📊 Calculate"])
        
        benchmark_df = pd.DataFrame(benchmarks)
        st.dataframe(benchmark_df)

# Export Results
st.subheader("📤 Export Results")
if st.button("Generate Optimization Report"):
    report_data = {
        "timestamp": pd.Timestamp.now(),
        "parameters": {
            "demand_factor": demand_factor,
            "seasonal_adjustment": seasonal_adjustment,
            "mumbai_capacity": capacity_w1,
            "kolkata_capacity": capacity_w2,
            "transport_cost_multiplier": transport_cost_multiplier,
            "inventory_cost_rate": inventory_cost_rate
        }
    }
    
    if 'total_cost' in locals():
        report_data["results"] = {
            "total_cost": total_cost,
            "total_production": total_production,
            "total_revenue": total_revenue,
            "profit": profit,
            "profit_margin": profit_margin,
            "avg_utilization": avg_utilization
        }
    
    # Convert to JSON for download
    import json
    report_json = json.dumps(report_data, indent=2, default=str)
    
    st.download_button(
        label="Download Optimization Report (JSON)",
        data=report_json,
        file_name=f"optimization_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )