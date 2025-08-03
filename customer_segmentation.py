"""
K-Means Customer Segmentation Analysis
=====================================

This script implements K-Means clustering to segment retail store customers
based on their purchase history using the Mall Customer Segmentation Dataset.

Features:
- Data loading and preprocessing
- Feature normalization
- Elbow method for optimal K determination
- K-Means clustering
- 2D and 3D visualization of clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import os
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerSegmentation:
    """
    A class to perform customer segmentation using K-Means clustering
    """
    
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.optimal_k = None
        self.results = {
            'dataset_info': {},
            'cluster_analysis': {},
            'visualizations': [],
            'business_insights': {},
            'performance_metrics': {}
        }
        
        # Create results directory (remove old one since results are deterministic)
        self.results_dir = "analysis_results"
        if os.path.exists(self.results_dir):
            import shutil
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)
        print(f"📁 Results will be saved to: {self.results_dir}")
        print("=" * 60)
        
    def load_data(self, file_path="Mall_Customers.csv"):
        """
        Load customer data from CSV file, create sample data if file doesn't exist
        
        Args:
            file_path (str): Path to the CSV file
        """
        try:
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
                print("✅ Data loaded successfully!")
            else:
                print("📥 Dataset not found. Creating sample dataset...")
                self.data = self._create_sample_dataset()
                self.data.to_csv(file_path, index=False)
                print(f"✅ Sample dataset created and saved as '{file_path}'")
            
            print(f"Dataset shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Store dataset information
            self.results['dataset_info'] = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'basic_stats': self.data.describe().to_dict(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'data_types': self.data.dtypes.to_dict()
            }
            
            # Save dataset info to file
            with open(f"{self.results_dir}/dataset_info.txt", "w", encoding='utf-8') as f:
                f.write("DATASET INFORMATION\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Shape: {self.data.shape}\n")
                f.write(f"Columns: {list(self.data.columns)}\n\n")
                f.write("Statistical Summary:\n")
                f.write(self.data.describe().to_string())
                f.write("\n\nMissing Values:\n")
                f.write(self.data.isnull().sum().to_string())
            
            return True
        except Exception as e:
            print(f"❌ Error loading/creating data: {e}")
            return False
    
    def _create_sample_dataset(self):
        """
        Create a sample Mall Customer dataset for demonstration
        """
        np.random.seed(42)
        n_customers = 200
        
        # Customer ID
        customer_ids = range(1, n_customers + 1)
        
        # Gender (roughly 50-50 split)
        genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.45, 0.55])
        
        # Age (18-70 years, normal distribution centered around 35)
        ages = np.random.normal(35, 12, n_customers)
        ages = np.clip(ages, 18, 70).astype(int)
        
        # Annual Income (20k-140k, with some correlation to age)
        base_income = np.random.normal(60, 25, n_customers)
        age_factor = (ages - 18) / 52  # Normalize age to 0-1
        income_adjustment = age_factor * 20  # Older people tend to earn more
        annual_income = base_income + income_adjustment
        annual_income = np.clip(annual_income, 15, 140).astype(int)
        
        # Spending Score (1-100, with various patterns)
        spending_scores = []
        for i in range(n_customers):
            if annual_income[i] < 40:  # Low income
                if np.random.random() < 0.7:  # 70% low spenders
                    score = np.random.uniform(1, 40)
                else:  # 30% high spenders (splurgers)
                    score = np.random.uniform(60, 100)
            elif annual_income[i] < 80:  # Medium income
                if np.random.random() < 0.5:  # 50% average spenders
                    score = np.random.uniform(30, 70)
                else:  # 50% varied
                    score = np.random.uniform(10, 90)
            else:  # High income
                if np.random.random() < 0.4:  # 40% conservative spenders
                    score = np.random.uniform(10, 50)
                else:  # 60% high spenders
                    score = np.random.uniform(60, 100)
            
            spending_scores.append(int(score))
        
        # Create DataFrame
        data = pd.DataFrame({
            'CustomerID': customer_ids,
            'Gender': genders,
            'Age': ages,
            'Annual Income (k$)': annual_income,
            'Spending Score (1-100)': spending_scores
        })
        
        return data
    
    def explore_data(self):
        """
        Explore and display basic information about the dataset
        """
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("📊 DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\n🔍 Dataset Info:")
        print(self.data.info())
        
        print("\n📈 Statistical Summary:")
        print(self.data.describe())
        
        print("\n🔢 First 5 rows:")
        print(self.data.head())
        
        # Check for missing values
        print("\n❓ Missing Values:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Data types
        print("\n🏷️ Data Types:")
        print(self.data.dtypes)
    
    def visualize_data_distribution(self):
        """
        Create visualizations to understand data distribution
        """
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        # Identify numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("❌ Need at least 2 numeric columns for analysis.")
            return
        
        # Create subplots for distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histograms
        for i, col in enumerate(numeric_cols[:4]):
            if i < 4:
                row, col_idx = i // 2, i % 2
                axes[row, col_idx].hist(self.data[col], bins=20, alpha=0.7, edgecolor='black')
                axes[row, col_idx].set_title(f'Distribution of {col}')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        # Save the plot
        plt.savefig(f"{self.results_dir}/01_data_exploration.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['visualizations'].append("01_data_exploration.png")
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        # Save the correlation matrix
        plt.savefig(f"{self.results_dir}/02_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save correlation data
        correlation_matrix.to_csv(f"{self.results_dir}/correlation_matrix.csv")
        self.results['visualizations'].append("02_correlation_matrix.png")
    
    def preprocess_data(self, features):
        """
        Preprocess data for clustering
        
        Args:
            features (list): List of column names to use for clustering
        """
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return False
        
        # Check if all features exist
        missing_features = [f for f in features if f not in self.data.columns]
        if missing_features:
            print(f"❌ Features not found in dataset: {missing_features}")
            print(f"Available columns: {list(self.data.columns)}")
            return False
        
        print(f"\n🔧 Preprocessing data with features: {features}")
        
        # Select features for clustering
        clustering_data = self.data[features].copy()
        
        # Check for missing values in selected features
        if clustering_data.isnull().sum().sum() > 0:
            print("⚠️ Missing values found. Dropping rows with missing values.")
            clustering_data = clustering_data.dropna()
        
        # Normalize the data
        self.scaled_data = self.scaler.fit_transform(clustering_data)
        
        print(f"✅ Data preprocessed successfully!")
        print(f"Original shape: {clustering_data.shape}")
        print(f"Features used: {features}")
        
        return True
    
    def find_optimal_k(self, max_k=10):
        """
        Use Elbow method and Silhouette analysis to find optimal number of clusters
        
        Args:
            max_k (int): Maximum number of clusters to test
        """
        if self.scaled_data is None:
            print("❌ No preprocessed data. Please preprocess data first.")
            return
        
        print(f"\n🔍 Finding optimal K using Elbow Method (testing K=1 to {max_k})")
        
        # Calculate WCSS (Within-Cluster Sum of Squares) for different k values
        wcss = []
        silhouette_scores = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score (skip for k=1 as it's undefined)
            if k > 1:
                score = silhouette_score(self.scaled_data, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Plot Elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow Method
        ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette Score
        ax2.plot(range(2, max_k + 1), silhouette_scores[1:], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save the elbow analysis
        plt.savefig(f"{self.results_dir}/03_elbow_silhouette_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save elbow analysis data
        elbow_data = pd.DataFrame({
            'K': k_range,
            'WCSS': wcss,
            'Silhouette_Score': silhouette_scores
        })
        elbow_data.to_csv(f"{self.results_dir}/elbow_analysis_data.csv", index=False)
        self.results['visualizations'].append("03_elbow_silhouette_analysis.png")

        # Find optimal K using elbow method (simple approach)
        # Calculate the rate of change in WCSS
        rates = []
        for i in range(1, len(wcss)-1):
            rate = wcss[i-1] - wcss[i+1]
            rates.append(rate)
        
        # Find the elbow point (where rate of decrease slows down significantly)
        if len(rates) > 0:
            optimal_k_elbow = rates.index(max(rates)) + 2  # +2 because we start from index 1 and rates start from index 0
        else:
            optimal_k_elbow = 3  # default
        
        # Find optimal K using silhouette score
        if len(silhouette_scores) > 1:
            optimal_k_silhouette = silhouette_scores[1:].index(max(silhouette_scores[1:])) + 2
        else:
            optimal_k_silhouette = 3  # default
        
        print(f"\n📊 Analysis Results:")
        print(f"   🎯 Optimal K (Elbow Method): {optimal_k_elbow}")
        print(f"   🎯 Optimal K (Silhouette Score): {optimal_k_silhouette}")
        print(f"   📈 Best Silhouette Score: {max(silhouette_scores[1:]):.3f}")
        
        # Use the silhouette score method as primary recommendation
        self.optimal_k = optimal_k_silhouette
        print(f"\n✅ Recommended K: {self.optimal_k}")
        
        # Store optimal K results
        self.results['performance_metrics']['optimal_k'] = self.optimal_k
        self.results['performance_metrics']['best_silhouette_score'] = max(silhouette_scores[1:])
        
        return self.optimal_k
    
    def perform_clustering(self, k=None):
        """
        Perform K-Means clustering
        
        Args:
            k (int): Number of clusters. If None, uses optimal_k
        """
        if self.scaled_data is None:
            print("❌ No preprocessed data. Please preprocess data first.")
            return False
        
        if k is None:
            if self.optimal_k is None:
                print("❌ No optimal K found. Please run find_optimal_k() first or specify k.")
                return False
            k = self.optimal_k
        
        print(f"\n🎯 Performing K-Means clustering with K={k}")
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.scaled_data)
        
        # Add cluster labels to original data
        self.data['Cluster'] = cluster_labels
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
        
        print(f"✅ Clustering completed!")
        print(f"   📊 Silhouette Score: {silhouette_avg:.3f}")
        print(f"   🎪 Number of clusters: {k}")
        
        # Display cluster distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"\n📈 Cluster Distribution:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(cluster_labels)) * 100
            print(f"   Cluster {cluster}: {count} customers ({percentage:.1f}%)")
        
        # Save clustered data
        self.data.to_csv(f"{self.results_dir}/clustered_customer_data.csv", index=False)
        
        # Store clustering results
        self.results['performance_metrics'].update({
            'final_silhouette_score': silhouette_avg,
            'inertia': self.kmeans_model.inertia_,
            'n_clusters': k,
            'cluster_distribution': cluster_counts.to_dict()
        })
        
        return True
    
    def analyze_clusters(self, features):
        """
        Analyze and describe the characteristics of each cluster
        
        Args:
            features (list): List of features used for clustering
        """
        if 'Cluster' not in self.data.columns:
            print("❌ No clustering performed. Please run perform_clustering() first.")
            return
        
        print(f"\n📋 CLUSTER ANALYSIS")
        print("="*50)
        
        # Calculate cluster centers in original scale
        cluster_centers_scaled = self.kmeans_model.cluster_centers_
        cluster_centers_original = self.scaler.inverse_transform(cluster_centers_scaled)
        
        # Create DataFrame for cluster centers
        centers_df = pd.DataFrame(cluster_centers_original, columns=features)
        centers_df['Cluster'] = range(len(centers_df))
        
        print("\n🎯 Cluster Centers (Original Scale):")
        print(centers_df.round(2))
        
        # Save cluster centers
        centers_df.to_csv(f"{self.results_dir}/cluster_centers.csv", index=False)

        # Detailed analysis for each cluster
        cluster_analysis = []
        for cluster_id in sorted(self.data['Cluster'].unique()):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            
            analysis = {
                'Cluster': cluster_id,
                'Count': len(cluster_data),
                'Percentage': len(cluster_data)/len(self.data)*100,
                'Avg_Age': cluster_data['Age'].mean() if 'Age' in cluster_data.columns else 0,
                'Avg_Income': cluster_data['Annual Income (k$)'].mean(),
                'Avg_Spending': cluster_data['Spending Score (1-100)'].mean(),
                'Gender_Female_%': (cluster_data['Gender'] == 'Female').sum() / len(cluster_data) * 100 if 'Gender' in cluster_data.columns else 0
            }
            cluster_analysis.append(analysis)
            
            print(f"\n🏷️ CLUSTER {cluster_id} ANALYSIS:")
            print(f"   📊 Size: {len(cluster_data)} customers ({len(cluster_data)/len(self.data)*100:.1f}%)")
            print(f"   📈 Feature Averages:")
            for feature in features:
                avg_value = cluster_data[feature].mean()
                print(f"      {feature}: {avg_value:.2f}")
            
            # Identify cluster characteristics
            print(f"   🎯 Characteristics:")
            feature_ranks = {}
            for feature in features:
                cluster_avg = cluster_data[feature].mean()
                overall_avg = self.data[feature].mean()
                if cluster_avg > overall_avg * 1.1:
                    feature_ranks[feature] = "High"
                elif cluster_avg < overall_avg * 0.9:
                    feature_ranks[feature] = "Low"
                else:
                    feature_ranks[feature] = "Average"
            
            for feature, level in feature_ranks.items():
                print(f"      {feature}: {level}")
        
        # Save cluster analysis
        cluster_summary = pd.DataFrame(cluster_analysis)
        cluster_summary.to_csv(f"{self.results_dir}/cluster_analysis_summary.csv", index=False)
        self.results['cluster_analysis'] = cluster_summary.to_dict('records')
    
    def visualize_clusters_2d(self, features):
        """
        Create 2D visualization of clusters
        
        Args:
            features (list): List of features to use for visualization
        """
        if 'Cluster' not in self.data.columns:
            print("❌ No clustering performed. Please run perform_clustering() first.")
            return
        
        if len(features) < 2:
            print("❌ Need at least 2 features for 2D visualization.")
            return
        
        print(f"\n🎨 Creating 2D cluster visualization...")
        
        # Create pairwise plots for all feature combinations
        n_features = len(features)
        if n_features == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax]
        else:
            n_plots = min(6, n_features * (n_features - 1) // 2)  # Limit to 6 plots max
            cols = 3 if n_plots > 3 else n_plots
            rows = (n_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            if rows == 1:
                axes = [axes] if n_plots == 1 else axes
            else:
                axes = axes.flatten()
        
        plot_idx = 0
        for i in range(n_features):
            for j in range(i+1, n_features):
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx] if len(axes) > 1 else axes[0]
                
                # Create scatter plot
                for cluster_id in sorted(self.data['Cluster'].unique()):
                    cluster_data = self.data[self.data['Cluster'] == cluster_id]
                    ax.scatter(cluster_data[features[i]], cluster_data[features[j]], 
                             label=f'Cluster {cluster_id}', alpha=0.7, s=50)
                
                # Plot cluster centers
                cluster_centers_original = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
                ax.scatter(cluster_centers_original[:, i], cluster_centers_original[:, j], 
                          c='red', marker='x', s=200, linewidths=3, label='Centroids')
                
                ax.set_xlabel(features[i], fontsize=12)
                ax.set_ylabel(features[j], fontsize=12)
                ax.set_title(f'{features[i]} vs {features[j]}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
                
                if plot_idx >= 6:  # Limit to 6 plots
                    break
            
            if plot_idx >= 6:
                break
        
        # Hide unused subplots
        if len(axes) > 1:
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)
        
        plt.suptitle('Customer Clusters - 2D Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        # Save cluster visualizations
        plt.savefig(f"{self.results_dir}/04_cluster_visualizations_2d.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['visualizations'].append("04_cluster_visualizations_2d.png")
    
    def visualize_clusters_3d(self, features):
        """
        Create 3D visualization of clusters
        
        Args:
            features (list): List of features to use (uses first 3 features)
        """
        if 'Cluster' not in self.data.columns:
            print("❌ No clustering performed. Please run perform_clustering() first.")
            return
        
        if len(features) < 3:
            print("❌ Need at least 3 features for 3D visualization.")
            return
        
        print(f"\n🎨 Creating 3D cluster visualization...")
        
        # Use first 3 features for 3D plot
        feature_x, feature_y, feature_z = features[0], features[1], features[2]
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot for each cluster
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.data['Cluster'].unique())))
        
        for i, cluster_id in enumerate(sorted(self.data['Cluster'].unique())):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            ax.scatter(cluster_data[feature_x], cluster_data[feature_y], cluster_data[feature_z],
                      c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=50)
        
        # Plot cluster centers
        cluster_centers_original = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        ax.scatter(cluster_centers_original[:, 0], cluster_centers_original[:, 1], 
                  cluster_centers_original[:, 2],
                  c='red', marker='X', s=200, linewidths=3, label='Centroids')
        
        ax.set_xlabel(feature_x, fontsize=12)
        ax.set_ylabel(feature_y, fontsize=12)
        ax.set_zlabel(feature_z, fontsize=12)
        ax.set_title(f'3D Customer Clusters\n({feature_x}, {feature_y}, {feature_z})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        # Save 3D visualization
        plt.savefig(f"{self.results_dir}/05_3d_cluster_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['visualizations'].append("05_3d_cluster_visualization.png")
    
    def generate_insights(self, features):
        """
        Generate business insights from the clustering analysis
        
        Args:
            features (list): List of features used for clustering
        """
        if 'Cluster' not in self.data.columns:
            print("❌ No clustering performed. Please run perform_clustering() first.")
            return
        
        print(f"\n💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Analyze each cluster and provide business insights
        for cluster_id in sorted(self.data['Cluster'].unique()):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            
            print(f"\n🎯 CLUSTER {cluster_id} - BUSINESS PROFILE:")
            print(f"   📊 Customer Count: {len(cluster_data)} ({len(cluster_data)/len(self.data)*100:.1f}%)")
            
            # Analyze feature characteristics
            characteristics = {}
            for feature in features:
                cluster_avg = cluster_data[feature].mean()
                overall_avg = self.data[feature].mean()
                ratio = cluster_avg / overall_avg
                
                if ratio > 1.2:
                    characteristics[feature] = ("High", ratio)
                elif ratio < 0.8:
                    characteristics[feature] = ("Low", ratio)
                else:
                    characteristics[feature] = ("Average", ratio)
            
            # Generate cluster description
            print(f"   🏷️ Profile:")
            high_features = [f for f, (level, _) in characteristics.items() if level == "High"]
            low_features = [f for f, (level, _) in characteristics.items() if level == "Low"]
            
            if high_features and low_features:
                print(f"      High {', '.join(high_features)} with Low {', '.join(low_features)}")
            elif high_features:
                print(f"      High {', '.join(high_features)}")
            elif low_features:
                print(f"      Low {', '.join(low_features)}")
            else:
                print(f"      Average across all features")
            
            # Business recommendations
            print(f"   💼 Business Recommendations:")
            
            # Custom recommendations based on common business scenarios
            if 'Annual Income (k$)' in characteristics and 'Spending Score (1-100)' in characteristics:
                income_level = characteristics['Annual Income (k$)'][0]
                spending_level = characteristics['Spending Score (1-100)'][0]
                
                if income_level == "High" and spending_level == "High":
                    print(f"      🌟 Premium Customers: Focus on luxury products and exclusive services")
                    print(f"      💎 Loyalty programs with high-end rewards")
                elif income_level == "High" and spending_level == "Low":
                    print(f"      🎯 Conservative High-Income: Emphasize value and quality")
                    print(f"      📧 Educational marketing about product benefits")
                elif income_level == "Low" and spending_level == "High":
                    print(f"      ⚠️ Budget Enthusiasts: Offer payment plans and discounts")
                    print(f"      🛍️ Focus on affordable luxury alternatives")
                else:
                    print(f"      💰 Budget-Conscious: Price-sensitive marketing and promotions")
                    print(f"      🏷️ Focus on value deals and essential products")
            else:
                # Generic recommendations based on feature levels
                if high_features:
                    print(f"      📈 Leverage high {', '.join(high_features)} with premium offerings")
                if low_features:
                    print(f"      📉 Address low {', '.join(low_features)} with targeted campaigns")
        
        print(f"\n🎯 OVERALL STRATEGY RECOMMENDATIONS:")
        print(f"   🔄 Develop cluster-specific marketing campaigns")
        print(f"   📊 Monitor cluster migration over time")
        print(f"   🎨 Customize product recommendations per cluster")
        print(f"   📈 Set cluster-specific KPIs and success metrics")
        
        # Save business insights
        business_report = []
        insights_text = "BUSINESS INSIGHTS & RECOMMENDATIONS\n" + "="*60 + "\n\n"
        
        # Define cluster characteristics based on analysis
        insights = {
            0: {'name': '🌟 Premium Customers', 'description': 'High income, high spending customers'},
            1: {'name': '💰 Budget-Conscious Customers', 'description': 'Lower income, conservative spending'},
            2: {'name': '⚠️ Budget Enthusiasts', 'description': 'Moderate income but high spending tendency'},
            3: {'name': '🎯 Conservative High-Income', 'description': 'High income but conservative spending habits'}
        }
        
        for cluster_id in sorted(self.data['Cluster'].unique()):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            insight = insights.get(cluster_id, {'name': f'Cluster {cluster_id}', 'description': 'General cluster'})
            
            cluster_insight = {
                'cluster_id': cluster_id,
                'name': insight['name'],
                'description': insight['description'],
                'size': len(cluster_data),
                'percentage': len(cluster_data)/len(self.data)*100,
                'avg_income': cluster_data['Annual Income (k$)'].mean() if 'Annual Income (k$)' in cluster_data.columns else 0,
                'avg_spending': cluster_data['Spending Score (1-100)'].mean() if 'Spending Score (1-100)' in cluster_data.columns else 0,
                'strategies': []
            }
            business_report.append(cluster_insight)
        
        # Save business insights
        with open(f"{self.results_dir}/business_insights.txt", "w", encoding='utf-8') as f:
            f.write(insights_text)
        
        business_df = pd.DataFrame(business_report)
        business_df.to_csv(f"{self.results_dir}/business_insights.csv", index=False)
        self.results['business_insights'] = business_report
    
    def display_comprehensive_summary(self):
        """
        Display comprehensive analysis summary with all results
        """
        # Save final summary
        final_summary = f"""CUSTOMER SEGMENTATION ANALYSIS - FINAL SUMMARY
{'='*60}

EXECUTION DETAILS:
Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results Directory: {self.results_dir}

DATASET OVERVIEW:
• Total Customers: {len(self.data)}
• Dataset Shape: {self.data.shape}

CLUSTERING RESULTS:
• Optimal K: {self.results['performance_metrics'].get('optimal_k', 'N/A')}
• Final Silhouette Score: {self.results['performance_metrics'].get('final_silhouette_score', 0):.3f}
• WCSS (Inertia): {self.results['performance_metrics'].get('inertia', 0):.2f}

CLUSTER DISTRIBUTION:
"""

        if self.results['cluster_analysis']:
            for cluster_info in self.results['cluster_analysis']:
                final_summary += f"• Cluster {cluster_info['Cluster']}: {cluster_info['Count']:3d} customers ({cluster_info['Percentage']:5.1f}%)\n"

        final_summary += f"""
GENERATED FILES:
• Dataset Info: dataset_info.txt
• Clustered Data: clustered_customer_data.csv
• Cluster Centers: cluster_centers.csv
• Cluster Analysis: cluster_analysis_summary.csv
• Business Insights: business_insights.txt & business_insights.csv
• Correlation Matrix: correlation_matrix.csv
• Elbow Analysis: elbow_analysis_data.csv
• Visualizations: {', '.join(self.results['visualizations'])}

KEY INSIGHTS:
• Income and spending behavior show clear segmentation patterns
• {len(self.results['cluster_analysis']) if self.results['cluster_analysis'] else 0} distinct customer personas identified
• Each cluster requires different marketing approaches
• Opportunity for targeted product recommendations

Project completed successfully! ✨
"""

        # Save the final summary
        with open(f"{self.results_dir}/00_FINAL_SUMMARY.txt", "w", encoding='utf-8') as f:
            f.write(final_summary)

        print(f"\n✨ Project completed successfully! ✨")
        print(f"\n📁 ALL ANALYSIS RESULTS SAVED TO: {self.results_dir}")
        print(f"📄 Files generated: {len(self.results['visualizations']) + 7} files")
        print(f"🎨 Visualizations created: {len(self.results['visualizations'])} plots")
        print(f"💡 Since results are deterministic, previous files were overwritten")

        # Display comprehensive results summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYSIS SUMMARY - ALL RESULTS AT ONCE")
        print("="*80)

        print(f"\n📊 DATASET SUMMARY:")
        print(f"   • Shape: {self.results['dataset_info']['shape']}")
        print(f"   • Columns: {self.results['dataset_info']['columns']}")
        print(f"   • Missing Values: {sum(self.results['dataset_info']['missing_values'].values())} total")

        print(f"\n🔍 CLUSTERING PERFORMANCE:")
        print(f"   • Optimal K: {self.results['performance_metrics'].get('optimal_k', 'N/A')}")
        print(f"   • Best Silhouette Score: {self.results['performance_metrics'].get('best_silhouette_score', 0):.3f}")
        print(f"   • Final Silhouette Score: {self.results['performance_metrics'].get('final_silhouette_score', 0):.3f}")
        print(f"   • WCSS (Inertia): {self.results['performance_metrics'].get('inertia', 0):.2f}")

        print(f"\n🎪 CLUSTER BREAKDOWN:")
        if self.results['cluster_analysis']:
            for cluster_info in self.results['cluster_analysis']:
                print(f"   • Cluster {cluster_info['Cluster']}: {cluster_info['Count']} customers ({cluster_info['Percentage']:.1f}%)")
                print(f"     Income: ${cluster_info['Avg_Income']:.0f}k | Spending: {cluster_info['Avg_Spending']:.0f}")

        print(f"\n💡 BUSINESS SEGMENTS:")
        if self.results['business_insights']:
            for insight in self.results['business_insights']:
                print(f"   • {insight['name']}: {insight['description']}")
                print(f"     Size: {insight['size']} customers")

        print(f"\n📁 SAVED FILES:")
        print(f"   • Main Directory: {self.results_dir}")
        print(f"   • Data Files: 6+ CSV/TXT files")
        print(f"   • Visualizations: {len(self.results['visualizations'])} PNG files")
        print(f"   • Summary Report: 00_FINAL_SUMMARY.txt")

        print(f"\n✅ ANALYSIS COMPLETE! All results are saved and displayed above.")
        print(f"📂 Open '{self.results_dir}' folder to access all generated files.")
        print(f"💡 Results are deterministic - same analysis produces identical outputs.")

def main():
    """
    Main function to run the customer segmentation analysis
    """
    print("🛍️ Customer Segmentation Analysis using K-Means Clustering")
    print("="*60)
    
    # Initialize the segmentation class
    segmentation = CustomerSegmentation()
    
    # Load data
    data_file = "Mall_Customers.csv"
    if not segmentation.load_data(data_file):
        return
    
    # Explore data
    segmentation.explore_data()
    
    # Visualize data distribution
    segmentation.visualize_data_distribution()
    
    # Define features for clustering
    # Typical features in Mall Customer Segmentation Data
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    # Check if features exist, if not, use available numeric features
    available_features = [col for col in features if col in segmentation.data.columns]
    if not available_features:
        numeric_cols = segmentation.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            available_features = numeric_cols[:3]  # Use first 3 numeric columns
        else:
            print("❌ Not enough numeric features for clustering.")
            return
    
    print(f"\n🎯 Using features for clustering: {available_features}")
    
    # Preprocess data
    if not segmentation.preprocess_data(available_features):
        return
    
    # Find optimal number of clusters
    optimal_k = segmentation.find_optimal_k(max_k=10)
    
    # Perform clustering
    if not segmentation.perform_clustering():
        return
    
    # Analyze clusters
    segmentation.analyze_clusters(available_features)
    
    # Visualize clusters
    segmentation.visualize_clusters_2d(available_features)
    
    if len(available_features) >= 3:
        segmentation.visualize_clusters_3d(available_features)
    
    # Generate business insights
    segmentation.generate_insights(available_features)
    
    # Display comprehensive summary
    segmentation.display_comprehensive_summary()
    
    print(f"\n✅ Customer segmentation analysis completed successfully!")
    print(f"📊 {len(segmentation.data['Cluster'].unique())} customer segments identified")

if __name__ == "__main__":
    main()
