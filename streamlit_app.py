import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Interactive Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .cluster-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        color: #262730;
    }
    
    .stSelectbox > div > div > div {
        color: #262730;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix multiselect visibility */
    .stMultiSelect > div > div {
        background-color: #ffffff;
        color: #262730;
    }
    
    .stMultiSelect [data-baseweb="select"] > div {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix number input visibility */
    .stNumberInput > div > div > input {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix slider labels */
    .stSlider > div > div > div > div {
        color: #262730 !important;
    }
    
    /* Ensure dropdown options are visible */
    .stSelectbox [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [role="option"] {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    /* Fix sidebar form elements */
    .sidebar .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def generate_synthetic_data(n_customers, income_range, age_range, spending_patterns, random_seed):
    """Generate synthetic customer data based on user inputs"""
    np.random.seed(random_seed)
    
    # Generate customer IDs
    customer_ids = range(1, n_customers + 1)
    
    # Generate genders
    genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.45, 0.55])
    
    # Generate ages based on user input
    ages = np.random.normal((age_range[0] + age_range[1]) / 2, 
                           (age_range[1] - age_range[0]) / 6, n_customers)
    ages = np.clip(ages, age_range[0], age_range[1]).astype(int)
    
    # Generate incomes based on user input
    base_income = np.random.normal((income_range[0] + income_range[1]) / 2, 
                                  (income_range[1] - income_range[0]) / 6, n_customers)
    incomes = np.clip(base_income, income_range[0], income_range[1])
    
    # Generate spending scores based on patterns
    spending_scores = []
    for income in incomes:
        if spending_patterns == "Realistic (Income-based)":
            # Realistic pattern: higher income tends to higher spending
            if income < income_range[0] + (income_range[1] - income_range[0]) * 0.4:
                score = np.random.uniform(10, 50)  # Low income, conservative spending
            elif income < income_range[0] + (income_range[1] - income_range[0]) * 0.7:
                score = np.random.uniform(30, 70)  # Medium income, varied spending
            else:
                score = np.random.uniform(50, 100)  # High income, higher spending
        elif spending_patterns == "Random":
            score = np.random.uniform(1, 100)
        elif spending_patterns == "Conservative":
            score = np.random.uniform(1, 50)
        elif spending_patterns == "High Spenders":
            score = np.random.uniform(50, 100)
        else:  # Mixed patterns
            if np.random.random() < 0.25:
                score = np.random.uniform(1, 30)   # 25% low spenders
            elif np.random.random() < 0.5:
                score = np.random.uniform(30, 70)  # 25% medium spenders
            else:
                score = np.random.uniform(70, 100) # 50% high spenders
            
        spending_scores.append(score)
    
    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (₹ Lakhs)': incomes,
        'Spending Score (1-100)': spending_scores
    })
    
    return data

def perform_clustering(data, features, n_clusters, random_state):
    """Perform K-Means clustering on the data"""
    # Prepare features for clustering
    X = data[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to data
    data_clustered = data.copy()
    data_clustered['Cluster'] = clusters
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, clusters)
    inertia = kmeans.inertia_
    
    # Get cluster centers in original scale
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_original, columns=features)
    centers_df['Cluster'] = range(n_clusters)
    
    return data_clustered, silhouette, inertia, centers_df, kmeans, scaler

def find_optimal_k(data, features, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette analysis"""
    X = data[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, clusters))
    
    return k_range, inertias, silhouette_scores

def create_cluster_visualization(data, features, centers_df):
    """Create interactive cluster visualizations"""
    if len(features) >= 2:
        # 2D scatter plot
        fig = px.scatter(data, 
                        x=features[0], 
                        y=features[1], 
                        color='Cluster',
                        title=f'Customer Segments - {features[0]} vs {features[1]}',
                        hover_data=['CustomerID', 'Gender', 'Age'],
                        color_discrete_sequence=px.colors.qualitative.Set3)
        
        # Add cluster centers
        fig.add_scatter(x=centers_df[features[0]], 
                       y=centers_df[features[1]],
                       mode='markers+text',
                       marker=dict(symbol='x', size=15, color='red', line=dict(width=2)),
                       text=[f'C{i}' for i in centers_df['Cluster']],
                       textposition="middle center",
                       name='Cluster Centers',
                       showlegend=True)
        
        fig.update_layout(height=500)
        return fig
    
    return None

def create_elbow_plot(k_range, inertias, silhouette_scores):
    """Create elbow method and silhouette analysis plots"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Elbow Method', 'Silhouette Analysis'))
    
    # Elbow plot
    fig.add_trace(go.Scatter(x=list(k_range), y=inertias, 
                            mode='lines+markers', name='WCSS',
                            line=dict(color='blue', width=3),
                            marker=dict(size=8)), row=1, col=1)
    
    # Silhouette plot
    fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, 
                            mode='lines+markers', name='Silhouette Score',
                            line=dict(color='red', width=3),
                            marker=dict(size=8)), row=1, col=2)
    
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Within-Cluster Sum of Squares", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def analyze_clusters(data, features, centers_df):
    """Analyze and describe each cluster"""
    cluster_analysis = []
    
    for cluster_id in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_percentage = (cluster_size / len(data)) * 100
        
        # Calculate feature means
        feature_means = cluster_data[features].mean()
        
        # Determine characteristics
        characteristics = {}
        for feature in features:
            mean_value = feature_means[feature]
            overall_mean = data[feature].mean()
            if mean_value > overall_mean * 1.1:
                characteristics[feature] = "High"
            elif mean_value < overall_mean * 0.9:
                characteristics[feature] = "Low"
            else:
                characteristics[feature] = "Medium"
        
        cluster_analysis.append({
            'Cluster': cluster_id,
            'Size': cluster_size,
            'Percentage': cluster_percentage,
            'Characteristics': characteristics,
            'Means': feature_means.to_dict()
        })
    
    return cluster_analysis

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛍️ Interactive Customer Segmentation Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for user inputs
    st.sidebar.markdown('<div class="sidebar-header">🎛️ Configuration Panel</div>', 
                       unsafe_allow_html=True)
    
    # Data Generation Parameters
    st.sidebar.subheader("📊 Dataset Configuration")
    
    n_customers = st.sidebar.slider("Number of Customers", 
                                   min_value=50, max_value=1000, 
                                   value=200, step=50,
                                   help="Total number of customers to generate")
    
    age_range = st.sidebar.slider("Age Range", 
                                 min_value=18, max_value=80, 
                                 value=(20, 70), step=1,
                                 help="Minimum and maximum age of customers")
    
    income_range = st.sidebar.slider("Annual Income Range (₹ Lakhs)", 
                                    min_value=2, max_value=50, 
                                    value=(3, 25), step=1,
                                    help="Minimum and maximum annual income in lakhs")
    
    spending_patterns = st.sidebar.selectbox("Spending Behavior Pattern",
                                           ["Realistic (Income-based)", "Random", 
                                            "Conservative", "High Spenders", "Mixed"],
                                           help="Choose how spending scores relate to income")
    
    # Add custom styling to ensure text is visible
    st.sidebar.markdown("""
    <style>
    .stSelectbox > label {
        color: #262730 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    random_seed = st.sidebar.number_input("Random Seed", 
                                         min_value=1, max_value=999, 
                                         value=42, step=1,
                                         help="Set seed for reproducible results")
    
    # Clustering Parameters
    st.sidebar.subheader("🎯 Clustering Configuration")
    
    available_features = ['Age', 'Annual Income (₹ Lakhs)', 'Spending Score (1-100)']
    selected_features = st.sidebar.multiselect("Features for Clustering",
                                              available_features,
                                              default=['Annual Income (₹ Lakhs)', 'Spending Score (1-100)'],
                                              help="Select which features to use for clustering")
    
    if len(selected_features) < 2:
        st.sidebar.error("Please select at least 2 features for clustering!")
        return
    
    auto_k = st.sidebar.checkbox("Auto-detect Optimal K", value=True,
                                help="Automatically find the best number of clusters")
    
    if not auto_k:
        n_clusters = st.sidebar.slider("Number of Clusters (K)", 
                                      min_value=2, max_value=10, 
                                      value=4, step=1,
                                      help="Manual selection of cluster count")
    
    # Analysis Options
    st.sidebar.subheader("📈 Analysis Options")
    
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    show_statistics = st.sidebar.checkbox("Show Statistical Summary", value=True)
    show_correlation = st.sidebar.checkbox("Show Correlation Matrix", value=True)
    
    # Generate button
    if st.sidebar.button("🚀 Generate Analysis", type="primary"):
        
        # Generate data
        with st.spinner("Generating synthetic customer data..."):
            data = generate_synthetic_data(n_customers, income_range, age_range, 
                                         spending_patterns, random_seed)
        
        # Store data in session state
        st.session_state['data'] = data
        st.session_state['selected_features'] = selected_features
        st.session_state['auto_k'] = auto_k
        if not auto_k:
            st.session_state['n_clusters'] = n_clusters
        st.session_state['show_raw_data'] = show_raw_data
        st.session_state['show_statistics'] = show_statistics
        st.session_state['show_correlation'] = show_correlation
    
    # Check if data exists in session state
    if 'data' not in st.session_state:
        st.info("👈 Configure your parameters in the sidebar and click 'Generate Analysis' to start!")
        
        # Show sample configuration
        st.subheader("🎯 What You Can Customize:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Dataset Parameters:**
            - Number of customers (50-1000)
            - Age range (18-80 years)
            - Income range ($15k-$200k)
            - Spending behavior patterns
            - Random seed for reproducibility
            """)
        
        with col2:
            st.markdown("""
            **🎯 Clustering Options:**
            - Choose features for analysis
            - Auto-detect optimal clusters
            - Manual cluster count selection
            - Advanced algorithm parameters
            """)
        
        with col3:
            st.markdown("""
            **📈 Visualization Controls:**
            - Interactive scatter plots
            - Elbow method analysis
            - Silhouette score evaluation
            - Detailed cluster profiling
            - Business insights generation
            """)
        
        return
    
    # Use data from session state
    data = st.session_state['data']
    selected_features = st.session_state['selected_features']
    auto_k = st.session_state['auto_k']
    show_raw_data = st.session_state['show_raw_data']
    show_statistics = st.session_state['show_statistics']
    show_correlation = st.session_state['show_correlation']
    
    # Main content area
    st.success(f"✅ Generated dataset with {len(data)} customers!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>👥 Customers</h3>
            <h2>{len(data):,}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        avg_age = data['Age'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>📅 Avg Age</h3>
            <h2>{avg_age:.1f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_income = data['Annual Income (₹ Lakhs)'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>💰 Avg Income</h3>
            <h2>₹{avg_income:.1f}L</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_spending = data['Spending Score (1-100)'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>🛒 Avg Spending</h3>
            <h2>{avg_spending:.0f}/100</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Data overview section
    if show_raw_data:
        st.subheader("📋 Raw Data Preview")
        st.dataframe(data.head(20), use_container_width=True)
    
    if show_statistics:
        st.subheader("📊 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    if show_correlation:
        st.subheader("🔗 Feature Correlation Matrix")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()
        
        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Heatmap",
                       color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering analysis
    st.subheader("🎯 K-Means Clustering Analysis")
    
    # Find optimal K if auto mode
    if auto_k:
        with st.spinner("Finding optimal number of clusters..."):
            k_range, inertias, silhouette_scores = find_optimal_k(data, selected_features)
            
            # Show elbow plot
            elbow_fig = create_elbow_plot(k_range, inertias, silhouette_scores)
            st.plotly_chart(elbow_fig, use_container_width=True)
            
            # Determine optimal K
            best_silhouette_idx = np.argmax(silhouette_scores)
            optimal_k = k_range[best_silhouette_idx]
            
            st.info(f"🎯 Recommended number of clusters: **{optimal_k}** (Silhouette Score: {silhouette_scores[best_silhouette_idx]:.3f})")
            n_clusters = optimal_k
    else:
        n_clusters = st.session_state['n_clusters']
    
    # Perform clustering
    with st.spinner("Performing K-Means clustering..."):
        data_clustered, silhouette, inertia, centers_df, kmeans_model, scaler = perform_clustering(
            data, selected_features, n_clusters, random_seed
        )
    
    # Clustering results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎪 Number of Clusters", n_clusters)
        st.metric("📈 Silhouette Score", f"{silhouette:.3f}")
    
    with col2:
        st.metric("🎯 WCSS (Inertia)", f"{inertia:.2f}")
        st.metric("✅ Algorithm", "K-Means")
    
    # Cluster visualization
    st.subheader("📊 Cluster Visualization")
    
    cluster_fig = create_cluster_visualization(data_clustered, selected_features, centers_df)
    if cluster_fig:
        st.plotly_chart(cluster_fig, use_container_width=True)
    
    # Cluster distribution
    st.subheader("📈 Cluster Distribution")
    cluster_counts = data_clustered['Cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(values=cluster_counts.values, 
                        names=[f'Cluster {i}' for i in cluster_counts.index],
                        title="Customer Distribution by Cluster")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(x=[f'Cluster {i}' for i in cluster_counts.index], 
                        y=cluster_counts.values,
                        title="Number of Customers per Cluster",
                        color=cluster_counts.values,
                        color_continuous_scale="viridis")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed cluster analysis
    st.subheader("🔍 Detailed Cluster Analysis")
    
    cluster_analysis = analyze_clusters(data_clustered, selected_features, centers_df)
    
    for analysis in cluster_analysis:
        with st.expander(f"🏷️ Cluster {analysis['Cluster']} - {analysis['Size']} customers ({analysis['Percentage']:.1f}%)"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Cluster Characteristics:**")
                for feature, level in analysis['Characteristics'].items():
                    emoji = "🔺" if level == "High" else "🔻" if level == "Low" else "➡️"
                    st.write(f"- {emoji} {feature}: {level}")
            
            with col2:
                st.write("**📈 Average Values:**")
                for feature, value in analysis['Means'].items():
                    if feature in ['Annual Income (₹ Lakhs)']:
                        st.write(f"- {feature}: ₹{value:.1f} Lakhs")
                    elif feature in ['Spending Score (1-100)']:
                        st.write(f"- {feature}: {value:.0f}/100")
                    else:
                        st.write(f"- {feature}: {value:.1f}")
            
            # Business recommendations
            st.write("**💼 Business Recommendations:**")
            
            characteristics = analysis['Characteristics']
            
            if 'Annual Income (₹ Lakhs)' in characteristics and 'Spending Score (1-100)' in characteristics:
                income_level = characteristics['Annual Income (₹ Lakhs)']
                spending_level = characteristics['Spending Score (1-100)']
                
                if income_level == "High" and spending_level == "High":
                    st.success("🌟 **Premium Customers**: Focus on luxury products, exclusive services, and premium loyalty programs")
                elif income_level == "Low" and spending_level == "Low":
                    st.info("💰 **Budget-Conscious**: Price-sensitive marketing, value deals, and essential products")
                elif income_level == "Low" and spending_level == "High":
                    st.warning("🎯 **Enthusiastic Spenders**: Target with attractive financing options and trendy products")
                elif income_level == "High" and spending_level == "Low":
                    st.warning("🎯 **Conservative High-Income**: Emphasize value, quality, and long-term benefits")
                else:
                    st.info("⚖️ **Balanced Customers**: Standard marketing approach with diverse product offerings")
    
    # Download results
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download clustered data
        csv_data = data_clustered.to_csv(index=False)
        st.download_button(
            label="📊 Download Clustered Data",
            data=csv_data,
            file_name=f"customer_segments_{n_clusters}clusters.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download cluster centers
        centers_csv = centers_df.to_csv(index=False)
        st.download_button(
            label="🎯 Download Cluster Centers",
            data=centers_csv,
            file_name=f"cluster_centers_{n_clusters}clusters.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download analysis report
        report = f"""
Customer Segmentation Analysis Report
=====================================

Dataset Configuration:
- Number of customers: {n_customers}
- Age range: {age_range[0]}-{age_range[1]} years
- Income range: ₹{income_range[0]}L-₹{income_range[1]}L
- Spending pattern: {spending_patterns}

Clustering Results:
- Number of clusters: {n_clusters}
- Silhouette score: {silhouette:.3f}
- WCSS (Inertia): {inertia:.2f}
- Features used: {', '.join(selected_features)}

Cluster Distribution:
"""
        for analysis in cluster_analysis:
            report += f"\nCluster {analysis['Cluster']}: {analysis['Size']} customers ({analysis['Percentage']:.1f}%)\n"
            for feature, level in analysis['Characteristics'].items():
                report += f"  - {feature}: {level}\n"
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name=f"segmentation_report_{n_clusters}clusters.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🛍️ Interactive Customer Segmentation Dashboard</p>
        <p>Built with Streamlit • Powered by Scikit-learn • Created for Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
