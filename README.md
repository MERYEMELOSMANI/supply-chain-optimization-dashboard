# Supply Chain Optimization Suite

A comprehensive supply chain optimization toolkit with multiple algorithms and interactive dashboard.

## 🚀 Features

### Optimization Algorithms
- **Linear Programming**: Classic optimization using PuLP
- **Genetic Algorithm**: Evolutionary optimization with Differential Evolution
- **Multi-objective Optimization**: Balancing cost, revenue, and utilization
- **Robust Optimization**: Handling uncertainty with multiple scenarios
- **Clustering-based Analysis**: Product grouping for strategic insights
- **Monte Carlo Simulation**: Risk analysis and uncertainty quantification
- **Sensitivity Analysis**: Parameter impact assessment

### Interactive Dashboard
- Real-time parameter adjustment
- Multiple visualization types
- Scenario comparison
- What-if analysis
- Risk assessment
- Performance benchmarking
- Export capabilities

## 📊 Data Structure

The system expects a CSV file (`supply_chain_data.csv`) with the following columns:
- `Product type`: Product categories (haircare, skincare, cosmetics)
- `Production volumes`: Production capacity
- `Manufacturing costs`: Cost per unit to manufacture
- `Costs`: Transportation/logistics costs
- `Price`: Selling price per unit
- `Revenue generated`: Total revenue
- `Location`: Warehouse locations
- Other optional columns for detailed analysis

## 🛠️ Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your data file (`supply_chain_data.csv`) is in the project directory.

## 🎯 Usage

### 1. Interactive Dashboard
Run the Streamlit dashboard for interactive optimization:
```bash
streamlit run optimization_dashboard.py
```

Features:
- Adjust parameters using sidebar controls
- View real-time optimization results
- Compare different scenarios
- Export results and reports
- Visualize allocation and utilization

### 2. Command Line Demo
Run the demonstration script to test all algorithms:
```bash
python demo_optimization.py
```

This will execute all optimization algorithms and display comparative results.

### 3. Python API
Use the optimization classes directly in your code:

```python
from advanced_optimization_algorithms import SupplyChainOptimizer
import pandas as pd

# Load data
data = pd.read_csv("supply_chain_data.csv")
optimizer = SupplyChainOptimizer(data)

# Run linear programming optimization
result = optimizer.linear_programming_optimization(
    demand_factor=0.8,
    capacities=[35000, 30000]
)

# Run genetic algorithm
ga_result = optimizer.genetic_algorithm_optimization(
    demand_factor=0.8,
    capacities=[35000, 30000],
    generations=100
)

# Perform clustering analysis
cluster_result = optimizer.clustering_based_optimization(n_clusters=3)
```

## 🔧 Configuration

### Dashboard Parameters
- **Demand Factor**: Overall market demand multiplier (0.1 - 2.0)
- **Seasonal Adjustment**: Seasonal demand variation (0.5 - 1.5)
- **Warehouse Capacities**: Mumbai and Kolkata capacity limits
- **Cost Parameters**: Transportation cost multipliers and inventory rates
- **Optimization Weights**: Balance between cost, revenue, and utilization

### Algorithm Parameters
- **Linear Programming**: Basic constraints and objective function
- **Genetic Algorithm**: Population size, generations, mutation rates
- **Multi-objective**: Weights for different objectives
- **Robust Optimization**: Scenario definitions and probabilities
- **Monte Carlo**: Number of simulations, parameter distributions

## 📈 Output Interpretation

### Key Metrics
- **Total Cost**: Combined manufacturing, transportation, and inventory costs
- **Total Production**: Sum of production across all warehouses
- **Total Revenue**: Revenue from all products
- **Profit**: Revenue minus total costs
- **Utilization**: Warehouse capacity utilization percentage

### Visualizations
- **Allocation Tables**: Production distribution by warehouse and product
- **Utilization Charts**: Warehouse capacity usage
- **Cost Breakdown**: Detailed cost component analysis
- **Product Distribution**: Production mix visualization
- **Scenario Comparison**: Multi-scenario analysis results

## 🎛️ Advanced Features

### Scenario Analysis
Compare multiple business scenarios:
- Conservative: Lower demand, higher costs
- Baseline: Current parameters
- Optimistic: Higher demand, lower costs
- Capacity Expansion: Increased warehouse capacity

### What-if Analysis
Interactive parameter modification to see immediate impact on:
- Total costs
- Production allocation
- Warehouse utilization
- Profitability

### Risk Assessment
Automated risk identification:
- High capacity utilization warnings
- Transportation cost risks
- Demand scenario risks
- Inventory cost alerts

### Performance Benchmarking
Compare results against industry standards:
- Profit margins
- Warehouse utilization rates
- Cost per unit metrics
- Inventory turnover ratios

## 🔄 Workflow

1. **Data Preparation**: Ensure CSV file is properly formatted
2. **Parameter Setting**: Configure optimization parameters
3. **Algorithm Selection**: Choose appropriate optimization method
4. **Execution**: Run optimization and analyze results
5. **Scenario Testing**: Compare multiple scenarios
6. **Decision Making**: Use insights for strategic planning

## 📋 File Structure

```
optimisation/
├── optimization_dashboard.py          # Interactive Streamlit dashboard
├── advanced_optimization_algorithms.py # Core optimization algorithms
├── demo_optimization.py               # Command-line demonstration
├── supply_chain_data.csv             # Input data file
├── requirements.txt                   # Python dependencies
├── README.md                         # This documentation
└── get-pip.py                        # Pip installer (if needed)
```

## 🛡️ Error Handling

The system includes comprehensive error handling for:
- Missing data files
- Invalid parameter values
- Infeasible optimization problems
- Numerical computation errors
- Visualization rendering issues

## 🎯 Best Practices

1. **Data Quality**: Ensure clean, consistent data
2. **Parameter Tuning**: Start with conservative parameters
3. **Scenario Testing**: Test multiple scenarios before decisions
4. **Regular Updates**: Update data and parameters regularly
5. **Result Validation**: Cross-check results with business logic

## 🔍 Troubleshooting

### Common Issues
- **No optimal solution**: Increase warehouse capacities or reduce demand
- **High costs**: Check transportation multipliers and inventory rates
- **Low utilization**: Adjust demand factors or consolidate operations
- **Visualization errors**: Ensure plotly is properly installed

### Performance Tips
- Use smaller datasets for testing
- Reduce Monte Carlo simulation runs for faster execution
- Adjust genetic algorithm parameters for speed vs. accuracy trade-off

## 🚀 Future Enhancements

Planned features:
- Multi-period optimization (temporal planning)
- Sustainability constraints (carbon footprint)
- Dynamic pricing optimization
- Supply chain disruption modeling
- Machine learning demand forecasting
- Real-time data integration

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Verify data file format and completeness
4. Ensure all dependencies are installed correctly

## 📄 License

This project is provided as-is for educational and commercial use. Please ensure proper attribution when using or modifying the code.
