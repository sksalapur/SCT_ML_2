# Customer Segmentation using K-Means Clustering 🛍️

This project implements a comprehensive K-Means clustering algorithm to segment retail store customers with an **interactive Streamlit web application**. Perfect for demonstrating machine learning capabilities with real-time user interaction.

## 🌐 **Interactive Web Dashboard**
🚀 **[Launch Live Demo](https://your-app-url.streamlit.app)** *(Deploy using instructions in DEPLOYMENT.md)*

### 🎛️ **User Input Features**
- **Dataset Configuration**: Adjust customer count, age range, income range, spending patterns
- **Clustering Parameters**: Select features, auto-detect optimal K, or set manually
- **Analysis Options**: Toggle data views, statistics, and correlation matrices
- **Real-time Generation**: Create custom synthetic datasets with configurable parameters
- **Export Results**: Download clustered data, cluster centers, and analysis reports

## 🎯 Project Overview

The goal is to identify distinct customer segments that can help businesses:
- Develop targeted marketing strategies
- Customize product recommendations
- Optimize customer experience
- Improve customer retention

## 📊 Features

### 🌐 **Interactive Streamlit Dashboard**
- **Real-time Parameter Control**: Sliders and inputs for dataset generation
- **Dynamic Visualizations**: Interactive Plotly charts with hover information
- **Flexible Feature Selection**: Choose which customer attributes to analyze
- **Automatic Optimization**: AI-powered optimal cluster detection
- **Business Intelligence**: Auto-generated insights and recommendations
- **Multi-format Export**: Download results as CSV, TXT, and analysis reports

### 🤖 **Core ML Features**
- **Auto-Saving Results**: All analysis results automatically saved to `analysis_results/` directory (overwrites previous runs since results are deterministic)
- **Data Loading & Exploration**: Comprehensive data analysis and visualization
- **Feature Normalization**: Standardization of numeric features for optimal clustering
- **Elbow Method**: Automatic determination of optimal number of clusters
- **Silhouette Analysis**: Validation of cluster quality
- **K-Means Clustering**: Implementation with customizable parameters
- **2D & 3D Visualizations**: Interactive cluster visualization with automatic saving
- **Business Insights**: Actionable recommendations for each customer segment
- **Comprehensive Summary**: Complete analysis overview displayed after each run

## 🛠️ Technologies Used

- **Python 3.13+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms

## 📁 Project Structure

```
📦 Customer Segmentation Project
├── 📄 customer_segmentation.py           # Main analysis script (includes data generation)
├── 📄 Customer_Segmentation_Analysis.ipynb # Interactive Jupyter notebook
├── 📄 Mall_Customers.csv                 # Customer dataset (auto-generated if missing)
├── 📄 requirements.txt                   # Python dependencies
└── 📄 README.md                          # Project documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Run the complete customer segmentation analysis
# The script will automatically create sample data if Mall_Customers.csv doesn't exist
python customer_segmentation.py
```

**Note**: The script automatically creates a sample dataset if the original Mall Customer data is not available. For the original dataset, you can download it from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and replace the generated file.

### 3. Access Results

After running the analysis, all results are automatically saved to a timestamped directory:
- **Directory Format**: `analysis_results_YYYYMMDD_HHMMSS/`
- **Files Generated**: 11+ files including visualizations, data exports, and comprehensive summary
- **Comprehensive Summary**: Displayed in terminal and saved as `00_FINAL_SUMMARY.txt`

## 📈 Analysis Steps

### 1. Data Exploration
- Load and inspect the customer dataset
- Check for missing values and data quality
- Generate statistical summaries
- Visualize data distributions

### 2. Data Preprocessing
- Select relevant features for clustering
- Normalize/standardize features using StandardScaler
- Handle missing values if any

### 3. Optimal Cluster Selection
- **Elbow Method**: Plot WCSS vs number of clusters
- **Silhouette Analysis**: Evaluate cluster separation quality
- Automatically determine optimal K value

### 4. K-Means Clustering
- Train K-Means model with optimal parameters
- Assign cluster labels to customers
- Calculate clustering performance metrics

### 5. Cluster Analysis
- Analyze characteristics of each cluster
- Calculate cluster centers and statistics
- Generate cluster profiles and descriptions

### 6. Visualization
- **2D Scatter Plots**: Feature pair visualizations
- **3D Scatter Plots**: Three-dimensional cluster view
- **Cluster Centers**: Centroid visualization
- **Distribution Plots**: Feature distribution analysis

### 7. Business Insights & Auto-Save
- Generate actionable business recommendations
- Identify target customer segments
- **Auto-save all results**: Visualizations, data exports, and analysis summaries
- **Comprehensive Summary**: Display complete analysis overview
- Suggest marketing strategies for each cluster

## 🎯 Customer Segments Typically Identified

Based on Annual Income and Spending Score:

1. **💎 High Income, High Spending** (Premium Customers)
   - Target with luxury products and exclusive services
   - Implement premium loyalty programs

2. **🎯 High Income, Low Spending** (Conservative Customers)
   - Focus on value proposition and quality
   - Educational marketing about product benefits

3. **⚠️ Low Income, High Spending** (Budget Enthusiasts)
   - Offer payment plans and attractive discounts
   - Focus on affordable luxury alternatives

4. **💰 Low Income, Low Spending** (Budget-Conscious)
   - Price-sensitive marketing and promotions
   - Emphasize value deals and essential products

## 📊 Key Metrics

- **Silhouette Score**: Measures cluster separation quality (higher is better)
- **Within-Cluster Sum of Squares (WCSS)**: Measures cluster compactness
- **Cluster Distribution**: Percentage of customers in each segment

## 🔧 Customization

### Modify Features
```python
# Edit the features list in customer_segmentation.py
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
```

### Adjust Clustering Parameters
```python
# Modify K-Means parameters
kmeans = KMeans(
    n_clusters=k, 
    random_state=42, 
    n_init=10,
    max_iter=300
)
```

### Custom Visualizations
```python
# Add custom plotting functions
def custom_visualization():
    # Your custom plotting code here
    pass
```

## 📊 Sample Output

```
🛍️ Customer Segmentation Analysis using K-Means Clustering
============================================================

✅ Data loaded successfully!
Dataset shape: (200, 5)
Columns: ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

🔧 Preprocessing data with features: ['Annual Income (k$)', 'Spending Score (1-100)']
✅ Data preprocessed successfully!

🔍 Finding optimal K using Elbow Method (testing K=1 to 10)
📊 Analysis Results:
   🎯 Optimal K (Elbow Method): 4
   🎯 Optimal K (Silhouette Score): 5
   📈 Best Silhouette Score: 0.553

🎯 Performing K-Means clustering with K=5
✅ Clustering completed!
   📊 Silhouette Score: 0.553
   🎪 Number of clusters: 5

📈 Cluster Distribution:
   Cluster 0: 43 customers (21.5%)
   Cluster 1: 38 customers (19.0%)
   Cluster 2: 41 customers (20.5%)
   Cluster 3: 39 customers (19.5%)
   Cluster 4: 39 customers (19.5%)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Data Source

- **Original Dataset**: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle
- **Sample Dataset**: Synthetically generated for demonstration purposes

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out!

---

**Happy Clustering! 🎉**
