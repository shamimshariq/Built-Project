#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\shami\OneDrive\Desktop\SEMESTER-II\Advance Machine Learning\Codes\8151csv_01d6ba1d74d6aeb0043e37dfcc20ea37_001\UKDA-8151-csv\csv\combined_hourly_data.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


data =df


# In[6]:


print(data.columns) 


# In[7]:


# Ensure the DataFrame is loaded as 'data'
# Calculate the approximate SPF for each entry
data['SPF_approx'] = (data['Hhp'] + data['Hhw']) / data['Ehp']

# Check the first few rows to verify the new column
print(data[['Hhp', 'Hhw', 'Ehp', 'SPF_approx']].head())


# In[8]:


import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Replace infinite values with NaN
data['SPF_approx'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Option 1: Fill NaN values with the mean or median of the column
data['SPF_approx'].fillna(data['SPF_approx'].mean(), inplace=True)

# Option 2: Remove rows with NaN values (if you prefer this approach)
# data.dropna(subset=['SPF_approx'], inplace=True)

# Now, retry standardizing and clustering
from sklearn.preprocessing import StandardScaler

# Assuming the SPF_approx column is now cleaned
spf_scaled = StandardScaler().fit_transform(data[['SPF_approx']].values)

# Proceed with K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(spf_scaled)

# You may proceed with the rest of the analysis as planned


# In[9]:


# Assuming 'spf_scaled' is your scaled data for clustering
wcss = []
for i in range(1, 11):  # Test 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(spf_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Determining Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(range(1, 11))

# Add a red vertical line at cluster 3
plt.axvline(x=3, color='red', linestyle='--', label='Optimal number of clusters = 3')
plt.legend()

plt.show()


# In[10]:


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# Assuming 'spf_scaled' is your scaled feature set
# Sample a subset of your data if it's large
if len(spf_scaled) > 10000:  # Arbitrary threshold
    np.random.seed(42)
    indices = np.random.choice(range(len(spf_scaled)), size=10000, replace=False)
    spf_scaled_sample = spf_scaled[indices]
else:
    spf_scaled_sample = spf_scaled

silhouette_coeffs = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(spf_scaled_sample)
    silhouette_avg = silhouette_score(spf_scaled_sample, cluster_labels)
    silhouette_coeffs.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_coeffs, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


# In[11]:


# Apply K-means clustering to categorize heat pumps into 3 performance categories
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(spf_scaled)

# To make clusters more interpretable, calculate the mean SPF for each cluster and sort them
cluster_means = data.groupby('KMeans_Cluster')['SPF_approx'].mean().sort_values().reset_index()
cluster_labels = {row['KMeans_Cluster']: 'Poorly Performing' if idx == 0 else 'Well Performing' if idx == len(cluster_means) - 1 else 'Average Performing' for idx, row in cluster_means.iterrows()}
data['Performance_Label_KMeans'] = data['KMeans_Cluster'].map(cluster_labels)


# In[12]:


import seaborn as sns
# Visualizing the results
plt.figure(figsize=(14, 6))

# K-means clusters
plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='SPF_approx', y=np.zeros(len(data)), hue='Performance_Label_KMeans', palette='viridis', s=50, alpha=0.6)
plt.title('K-means Clustering of Heat Pumps')
plt.xlabel('Approximate SPF')
plt.yticks([])


# In[13]:


print(data.columns) 


# In[14]:


# Calculate a simple moving average of the SPF_approx values
data_sorted = data.sort_values('SPF_approx').reset_index()
window_size = 50  # Change based on your dataset size and desired smoothness
data_sorted['SPF_moving_avg'] = data_sorted['SPF_approx'].rolling(window=window_size).mean()

# Plotting the moving average
plt.figure(figsize=(10, 6))
plt.plot(data_sorted['SPF_approx'], data_sorted['SPF_moving_avg'], label='Moving Average of SPF')
plt.title('Moving Average of Approximate SPF')
plt.xlabel('Approximate SPF')
plt.ylabel('Moving Average of SPF')
plt.legend()
plt.show()


# In[15]:


# Define seasons based on months
def season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# Apply the season function to the Month column
data['Season'] = data['Month'].apply(season)

# Group the data by Season and calculate the average SPF for each season
seasonal_spf = data.groupby('Season')['SPF_approx'].mean().reset_index()

seasonal_spf


# In[16]:


print(data.columns) 


# In[17]:


print(data.head())


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Check the distribution of SPF_approx values
plt.figure(figsize=(10, 6))
sns.histplot(data['SPF_approx'], bins=50, kde=True)
plt.title('Distribution of Approximated SPF Values')
plt.xlabel('SPF Approximation')
plt.ylabel('Frequency')
plt.show()


# In[19]:


# Recalculating the SPF_log column
data['SPF_log'] = np.log(data['SPF_approx'] + 0.001)  # Adding a small constant to avoid log(0)

# Recalculate the mean log SPF for each cluster (assuming clusters are already defined)
cluster_mean_spf = data.groupby('KMeans_Cluster')['SPF_log'].mean().sort_values().reset_index()

# Mapping the clusters back to their original SPF scale for interpretation
cluster_mean_spf['Mean_Spf'] = np.exp(cluster_mean_spf['SPF_log']) - 0.001  # Subtracting the small constant added earlier

# Assign labels based on the sorted order (ascending SPF values)
cluster_labels = {row['KMeans_Cluster']: 'Poorly Performing' if idx == 0 else 'Well Performing' if idx == len(cluster_mean_spf) - 1 else 'Average Performing' for idx, row in cluster_mean_spf.iterrows()}
data['Performance_Label'] = data['KMeans_Cluster'].map(cluster_labels)

# Summary of clusters and their interpreted labels
cluster_summary = data.groupby('Performance_Label')['SPF_log'].agg(['mean', 'std', 'count'])
cluster_summary['Mean_Spf'] = np.exp(cluster_summary['mean']) - 0.001
cluster_summary.drop(['mean', 'std'], axis=1, inplace=True)

cluster_summary.reset_index(), cluster_labels


# In[20]:


# Analyzing the characteristics of each performance group
# Focusing on temperatures and electricity consumption metrics
features = ['Tco', 'Tin', 'Tsf', 'Twf','Hhp','Hhw', 'Ehp', 'Edhw', 'Esp', 'Eboost', 'Performance_Label']
cluster_features_summary = data[features].groupby('Performance_Label').mean().reset_index()

cluster_features_summary


# In[21]:


# Example of how to segment your data by seasons
# This assumes you have a 'Season' column in your 'data' DataFrame to categorize each row into a season

seasonal_data = data.groupby('Season').apply(lambda x: x).to_dict('index')

# Alternatively, if you need to manually define seasons based on month or other criteria:
seasonal_data = {
    'Winter': data[data['Month'].isin([12, 1, 2])],
    'Spring': data[data['Month'].isin([3, 4, 5])],
    'Summer': data[data['Month'].isin([6, 7, 8])],
    'Autumn': data[data['Month'].isin([9, 10, 11])]
}

# Note: The above method assumes you have a 'Month' column to determine the season.
# Adjust the criteria as necessary based on your dataset's structure and the definition of seasons.


# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming 'SPF_log' values exist in your dataset and are representative for scaling
scaler = StandardScaler().fit(data['SPF_log'].values.reshape(-1, 1))

seasonal_clusters = {}
for season, subset in seasonal_data.items():
    # Transform the SPF_log values for the current season using the fitted scaler
    spf_scaled_season = scaler.transform(subset['SPF_log'].values.reshape(-1, 1))
    
    # K-means Clustering
    kmeans_season = KMeans(n_clusters=3, random_state=42)
    seasonal_clusters[season] = subset.copy()
    seasonal_clusters[season]['KMeans_Cluster'] = kmeans_season.fit_predict(spf_scaled_season)


# In[23]:


import numpy as np

# Overall Performance Clusters: Average SPF Values
overall_spf_means = data.groupby('Performance_Label')['SPF_approx'].mean().reset_index()

# Plot
plt.figure(figsize=(6, 6))
sns.barplot(x='Performance_Label', y='SPF_approx', data=overall_spf_means, order=['Poorly Performing', 'Average Performing', 'Well Performing'], palette='viridis')
plt.title('Average SPF Values by Performance Cluster')
plt.xlabel('Performance Cluster')
plt.ylabel('Average SPF Value')
plt.yscale('log')
plt.xticks(rotation=45)
plt.show()


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

# Selecting key features for the box plots
features_for_plots = ['Tco', 'Tin','Twf','Ehp', 'Hhp']

# Creating box plots for each feature across the performance clusters
# Adjust the figsize to make the figure wider and the layout 2 rows x 3 columns to accommodate all plots
plt.figure(figsize=(20, 15))  # Increase the figure size for better visibility and to fit two rows

for i, feature in enumerate(features_for_plots):
    plt.subplot(2, 3, i + 1)  # Adjust the subplot grid to 2 rows x 3 columns
    sns.boxplot(x='Performance_Label', y=feature, data=data, order=['Poorly Performing', 'Average Performing', 'Well Performing'])
    plt.title(f'Distribution of {feature} by Performance Cluster')
    plt.xlabel('Performance Cluster')
    plt.ylabel(feature)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[28]:


# Calculating average SPF values for clusters within the Winter season
winter_spf_means = seasonal_clusters['Winter'].groupby('KMeans_Cluster')['SPF_approx'].mean().reset_index()

# Plot
plt.figure(figsize=(6, 6))
sns.barplot(x='KMeans_Cluster', y='SPF_approx', data=winter_spf_means, palette='coolwarm')
plt.title('Average SPF Values by Cluster in Winter')
plt.xlabel('Cluster')
plt.ylabel('Average SPF Value')
plt.yscale('log')
plt.xticks(np.arange(3), ['Poorly Performing', 'Average Performing', 'Well Performing'])
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming features_for_plots and seasonal_clusters['Winter'] are predefined
features_for_plots = ['Tco', 'Tin','Twf', 'Ehp', 'Hhp']

# Creating box plots for each feature across the clusters in Winter
plt.figure(figsize=(20, 10))  # Adjusted figure size for better visibility and to accommodate two rows

for i, feature in enumerate(features_for_plots):
    plt.subplot(2, 3, i + 1)  # Adjusting the subplot to 2 rows x 3 columns grid
    sns.boxplot(x='KMeans_Cluster', y=feature, data=seasonal_clusters['Winter'], palette='coolwarm')
    plt.title(f'Winter: Distribution of {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.xticks(np.arange(3), ['Poorly Performing', 'Average Performing', 'Well Performing'])
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming features_for_plots and seasonal_clusters['Winter'] are predefined
features_for_plots = ['Tco', 'Tin','Twf', 'Ehp', 'Hhp']

# Creating box plots for each feature across the clusters in Summer
plt.figure(figsize=(20, 10))  # Adjusted figure size for better visibility and to accommodate two rows

for i, feature in enumerate(features_for_plots):
    plt.subplot(2, 3, i + 1)  # Adjusting the subplot to 2 rows x 3 columns grid
    sns.boxplot(x='KMeans_Cluster', y=feature, data=seasonal_clusters['Summer'], palette='coolwarm')
    plt.title(f'Summer: Distribution of {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.xticks(np.arange(3), ['Poorly Performing', 'Average Performing', 'Well Performing'])
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming features_for_plots and seasonal_clusters['Winter'] are predefined
features_for_plots = ['Tco', 'Tin','Twf', 'Ehp', 'Hhp']

# Creating box plots for each feature across the clusters in Spring
plt.figure(figsize=(20, 10))  # Adjusted figure size for better visibility and to accommodate two rows

for i, feature in enumerate(features_for_plots):
    plt.subplot(2, 3, i + 1)  # Adjusting the subplot to 2 rows x 3 columns grid
    sns.boxplot(x='KMeans_Cluster', y=feature, data=seasonal_clusters['Spring'], palette='coolwarm')
    plt.title(f'Spring: Distribution of {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.xticks(np.arange(3), ['Poorly Performing', 'Average Performing', 'Well Performing'])
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming features_for_plots and seasonal_clusters['Winter'] are predefined
features_for_plots = ['Tco', 'Tin','Twf', 'Ehp', 'Hhp']

# Creating box plots for each feature across the clusters in Autumn
plt.figure(figsize=(20, 10))  # Adjusted figure size for better visibility and to accommodate two rows

for i, feature in enumerate(features_for_plots):
    plt.subplot(2, 3, i + 1)  # Adjusting the subplot to 2 rows x 3 columns grid
    sns.boxplot(x='KMeans_Cluster', y=feature, data=seasonal_clusters['Autumn'], palette='coolwarm')
    plt.title(f'Autumn: Distribution of {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.xticks(np.arange(3), ['Poorly Performing', 'Average Performing', 'Well Performing'])
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# In[33]:


# Example: Preparing the 'seasonal_averages' DataFrame
import pandas as pd

# Assuming 'data' is your existing DataFrame and includes a 'Season' column along with the metrics
# Calculate the mean of each metric grouped by 'Season'
seasonal_averages = data.groupby('Season')[['SPF_approx', 'Tco', 'Tin', 'Twf', 'Ehp','Hhp']].mean().reset_index()

# Now, with 'seasonal_averages' prepared, you can proceed with the plotting code as you've written.


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'seasonal_averages' contains 'Hhp' with valid data
# fig, axs setup is correctly sized for 6 plots (3 rows, 2 columns)
fig, axs = plt.subplots(2, 3, figsize=(15, 15))

# Now correctly lists 6 metrics
seasonal_metrics = ['SPF_approx', 'Tco', 'Tin', 'Twf', 'Ehp', 'Hhp']
titles = ['Average SPF by Season', 'Average Coolant Outlet Temperature (Tco) by Season',
          'Average Inlet Temperature (Tin) by Season', 'Average Water Flow Temperature (Twf) by Season',
          'Average Electricity Consumption (Ehp) by Season', 'Average Heat Output (Hhp) by Season']
y_labels = ['Average SPF', 'Tco (°C)', 'Tin (°C)', 'Twf (°C)', 'Ehp (kWh)', 'Hhp (kWh)']

axs_flat = axs.flatten()

for i, metric in enumerate(seasonal_metrics):
    sns.barplot(x='Season', y=metric, data=seasonal_averages, ax=axs_flat[i], palette='cubehelix')
    axs_flat[i].set_title(titles[i])
    axs_flat[i].set_xlabel('Season')
    axs_flat[i].set_ylabel(y_labels[i])

# Now correctly plots all 6 metrics without needing to remove any subplot
plt.tight_layout()
plt.show()


# In[39]:


# Define time-of-day categories
def time_of_day(hour):
    if 5 <= hour <= 11:
        return 'Morning'
    elif 12 <= hour <= 16:
        return 'Afternoon'
    elif 17 <= hour <= 20:
        return 'Evening'
    else:
        return 'Night'

data['TimeOfDay'] = data['Hour'].apply(time_of_day)

# Aggregating average values for each time-of-day category
time_of_day_averages = data.groupby('TimeOfDay').agg({'SPF_approx': 'mean', 'Tco': 'mean', 'Tin': 'mean', 'Ehp': 'mean'}).reset_index()

# Visualizing the time-of-day averages for SPF, Tco, Tin, and Ehp
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
time_of_day_metrics = ['SPF_approx', 'Tco', 'Tin', 'Ehp']
titles = ['Average SPF by Time of Day', 'Average Coolant Outlet Temperature (Tco) by Time of Day',
          'Average Inlet Temperature (Tin) by Time of Day', 'Average Electricity Consumption (Ehp) by Time of Day']
y_labels = ['Average SPF', 'Tco (°C)', 'Tin (°C)', 'Ehp (kWh)']

for i, ax in enumerate(axs.flatten()):
    sns.barplot(x='TimeOfDay', y=time_of_day_metrics[i], data=time_of_day_averages, ax=ax, palette='viridis', order=['Morning', 'Afternoon', 'Evening', 'Night'])
    ax.set_title(titles[i])
    ax.set_xlabel('Time of Day')
    ax.set_ylabel(y_labels[i])

plt.tight_layout()
plt.show()


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns

# Aggregating data into hourly averages
hourly_averages = data.groupby('Hour').agg({'SPF_approx': 'mean', 'Tco': 'mean', 'Tin': 'mean', 'Twf': 'mean', 'Ehp': 'mean','Hhp':'mean'}).reset_index()

# Plotting line graphs for SPF, Tco, Tin, Twf, and Ehp over the course of a day
fig, axs = plt.subplots(3, 2, figsize=(18, 12))  # Maintain the 3x2 grid for 5 plots
variables = ['SPF_approx', 'Tco', 'Tin', 'Twf', 'Ehp','Hhp']
titles = ['Hourly Average SPF', 'Hourly Average Coolant Outlet Temperature (Tco)',
          'Hourly Average Inlet Temperature (Tin)', 'Hourly Average Water Flow Temperature (Twf)', 'Hourly Average Electricity Consumption (Ehp)','Hourly Average Heat Output (Hhp)']
y_labels = ['Average SPF', 'Tco (°C)', 'Tin (°C)', 'Twf (°C)', 'Ehp (kWh)','Hhp(kWh)']

# Flatten the axs array for easier indexing and loop through the number of variables
axs_flat = axs.flatten()

for i, variable in enumerate(variables):
    axs_flat[i].plot(hourly_averages['Hour'], hourly_averages[variable], marker='o', linestyle='-', color='tab:blue')
    axs_flat[i].set_title(titles[i])
    axs_flat[i].set_xlabel('Hour of the Day')
    axs_flat[i].set_ylabel(y_labels[i])
    axs_flat[i].grid(True)


plt.tight_layout()
plt.show()


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'hourly_cluster_averages_vars' has been properly created
hourly_cluster_averages_vars = data.groupby(['Hour', 'Performance_Label'])[variables_to_plot].mean().reset_index()

variables_to_plot = ['Tco', 'Tin', 'Twf', 'Ehp', 'Hhp']
n_vars = len(variables_to_plot)  # Number of variables you have
n_cols = 2  # You want 2 columns
n_rows = n_vars // n_cols + (n_vars % n_cols > 0)  # This will determine the number of rows you need

fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 18), squeeze=False)  # squeeze=False parameter makes sure axs is always a 2D array
axs = axs.flatten()  # Flatten to 1D array for easier indexing

for i, var in enumerate(variables_to_plot):
    sns.lineplot(data=hourly_cluster_averages_vars, x='Hour', y=var, hue='Performance_Label', style='Performance_Label', markers=True, dashes=False, palette='viridis', ax=axs[i])
    axs[i].set_title(f'Hourly Average {var} by Performance Cluster')
    axs[i].set_xlabel('Hour of the Day')
    axs[i].set_ylabel(var)
    axs[i].legend(title='Performance Cluster')
    axs[i].grid(True)

# Hide any unused subplots
for j in range(n_vars, n_rows * n_cols):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


# In[80]:


from scipy import stats

# Correcting the column name to 'KMeans_Cluster' or whatever your actual cluster label column is named
anova_result = stats.f_oneway(data[data['KMeans_Cluster'] == 0]['SPF_approx'],
                              data[data['KMeans_Cluster'] == 1]['SPF_approx'],
                              data[data['KMeans_Cluster'] == 2]['SPF_approx'])

print(f"ANOVA Result for SPF_approx: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")


# In[81]:


from sklearn.cluster import MiniBatchKMeans

# Use MiniBatchKMeans to reduce the dataset size
n_pre_clusters = 1000  # Adjust based on your dataset and memory capacity
mbk = MiniBatchKMeans(n_clusters=n_pre_clusters, batch_size=100)
pre_labels = mbk.fit_predict(spf_scaled)

# Now use the cluster centers for hierarchical clustering
centers = mbk.cluster_centers_


# In[83]:


from sklearn.cluster import AgglomerativeClustering

# Assuming `centers` are the cluster centers from MiniBatchKMeans
# Now apply AgglomerativeClustering to these centers

# Initialize AgglomerativeClustering
agglo_clustering = AgglomerativeClustering(n_clusters=3,  # or any other desired number of clusters
                                           affinity='euclidean',  # Distance measure
                                           linkage='ward')  # Linkage criterion

# Fit model to the cluster centers
agglo_labels = agglo_clustering.fit_predict(centers)

# Optional: Map the initial cluster labels (from MiniBatchKMeans) to the final hierarchical cluster labels
final_labels = np.copy(pre_labels)
for original_label, new_label in enumerate(agglo_labels):
    final_labels[pre_labels == original_label] = new_label

# At this point, `final_labels` contains the hierarchical cluster assignments for each point in the original dataset


# In[85]:


# Example step where final_labels would be created (ensure this step was executed correctly)
data['final_labels'] = final_labels  # This assumes 'final_labels' variable was defined earlier as shown in previous steps


# In[86]:


import numpy as np
import pandas as pd

# Perform the aggregation again with correct datatypes
cluster_characteristics = data.groupby('final_labels').agg({
    'Tco': ['mean', 'median', 'std'],
    'Tin': ['mean', 'median', 'std'],
    'Twf': ['mean', 'median', 'std'],
    'Ehp': ['mean', 'median', 'std']
})

# Display the aggregated DataFrame
print(cluster_characteristics)


# In[87]:


from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform hierarchical clustering on the cluster centers from MiniBatchKMeans
Z_centers = linkage(centers, 'ward')

# Plot the dendrogram for the cluster centers
plt.figure(figsize=(12, 7))
dendrogram(Z_centers, leaf_rotation=90., leaf_font_size=8., color_threshold=0)
plt.title('Dendrogram for Cluster Centers')
plt.xlabel('Cluster center index')
plt.ylabel('Distance')
plt.show()


# In[88]:


plt.figure(figsize=(20, 10))  # Increase figure size
dendrogram(Z_centers, leaf_rotation=90., leaf_font_size=10.)  # Adjust font size
plt.title('Dendrogram for Cluster Centers')
plt.xlabel('Cluster center index')
plt.ylabel('Distance')
plt.show()


# In[89]:


import plotly.figure_factory as ff

fig = ff.create_dendrogram(Z_centers, orientation='top')
fig.update_layout(width=800, height=600)
fig.show()


# In[90]:


plt.figure(figsize=(20, 10))
dendrogram(Z_centers, truncate_mode='lastp', p=30,  # Show only the last p merged clusters
           leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title('Dendrogram for Cluster Centers')
plt.xlabel('Cluster center index')
plt.ylabel('Distance')
plt.show()


# In[92]:


from sklearn.decomposition import PCA


# Proceed with adding PCA results back to your dataframe or further analysis
from sklearn.impute import SimpleImputer

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Perform imputation on the selected columns
imputed_data = imputer.fit_transform(data[['Tco', 'Tin', 'Twf', 'Ehp']])

# Perform PCA on the imputed data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(imputed_data)

# Add PCA results back to the dataframe
data['pca_one'] = pca_result[:, 0]
data['pca_two'] = pca_result[:, 1]

# Visualization
plt.figure(figsize=(16,10))
sns.scatterplot(x="pca_one", y="pca_two", hue=data['final_labels'], palette=sns.color_palette("hsv", 3), data=data)
plt.title('PCA Result colored by cluster label')
plt.show()


# In[ ]:


import numpy as np

# Count the number of points in each cluster
unique, counts = np.unique(final_labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))
print(cluster_sizes)


# In[25]:


# Assuming 'data' is your DataFrame and it includes 'SPF_approx', 'Tco', 'Tin', 'Twf', and 'Cluster'
plt.figure(figsize=(18, 6))

# Plot for Tco vs. SPF_approx
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st plot
sns.scatterplot(x='Tco', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Coolant Outlet Temperature (Tco)')
plt.xlabel('Coolant Outlet Temperature (Tco)')
plt.ylabel('SPF Approximation')

# Plot for Tin vs. SPF_approx
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd plot
sns.scatterplot(x='Tin', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Inlet Temperature (Tin)')
plt.xlabel('Inlet Temperature (Tin)')
# Hide the y-axis label to avoid clutter
plt.ylabel('')

# Plot for Twf vs. SPF_approx
plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd plot
sns.scatterplot(x='Twf', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Water Flow Temperature (Twf)')
plt.xlabel('Water Flow Temperature (Twf)')
# Hide the y-axis label to avoid clutter
plt.ylabel('')

# Plot for Ehp vs. SPF_approx
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 4th plot
sns.scatterplot(x='Ehp', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs.Hourly Average Electricity Consumption (Ehp)')
plt.xlabel('Hourly Average Electricity Consumption (Ehp)')
plt.ylabel('SPF Approximation')

# Plot for Hhp vs. SPF_approx
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 5th plot
sns.scatterplot(x='Hhp', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Hourly Average Heat Output (Hhp)')
plt.xlabel('Hourly Average Heat Output (Hhp)')
plt.ylabel('SPF Approximation')

plt.tight_layout()
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame and it includes 'SPF_approx', 'Tco', 'Tin', 'Twf', 'Ehp', and 'Hhp'
plt.figure(figsize=(18, 10))  # Adjusted for more vertical space

# Plot for Tco vs. SPF_approx
plt.subplot(2, 3, 1)  # 2 rows, 3 columns, 1st plot
sns.scatterplot(x='Tco', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Coolant Outlet Temperature (Tco)')
plt.xlabel('Coolant Outlet Temperature (Tco)')
plt.ylabel('SPF Approximation')

# Plot for Tin vs. SPF_approx
plt.subplot(2, 3, 2)  # 2 rows, 3 columns, 2nd plot
sns.scatterplot(x='Tin', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Inlet Temperature (Tin)')
plt.xlabel('Inlet Temperature (Tin)')
plt.ylabel('')  # Optionally hide for clarity if labels are repetitive

# Plot for Twf vs. SPF_approx
plt.subplot(2, 3, 3)  # 2 rows, 3 columns, 3rd plot
sns.scatterplot(x='Twf', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Water Flow Temperature (Twf)')
plt.xlabel('Water Flow Temperature (Twf)')
plt.ylabel('')  # Optionally hide for clarity if labels are repetitive

# Plot for Ehp vs. SPF_approx
plt.subplot(2, 3, 4)  # 2 rows, 3 columns, 4th plot
sns.scatterplot(x='Ehp', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Electricity Consumption (Ehp)')
plt.xlabel('Electricity Consumption (Ehp)')
plt.ylabel('SPF Approximation')

# Plot for Hhp vs. SPF_approx
plt.subplot(2, 3, 5)  # 2 rows, 3 columns, 5th plot
sns.scatterplot(x='Hhp', y='SPF_approx', hue='Cluster', data=data, palette='viridis')
plt.title('SPF Approximation vs. Heat Output (Hhp)')
plt.xlabel('Heat Output (Hhp)')
plt.ylabel('SPF Approximation')

plt.tight_layout()
plt.show()


# In[94]:


import pandas as pd

# Placeholder for the actual DataFrame and column name
# Ensure 'your_column_name' matches the name of the column in your DataFrame containing the SPF values
your_column_name = 'SPF_approx'  # Replace 'SPF_approx' with your actual column name if different

# Define the ranges
ranges = [0, 2, 5, 10, float('inf')]
range_labels = ['0-2', '2-5', '5-10', '>10']

# Use pd.cut to categorize the SPF values into the defined ranges
data['SPF_Range'] = pd.cut(data[your_column_name], bins=ranges, labels=range_labels, right=False)

# Count the number of values in each range
spf_range_counts = data['SPF_Range'].value_counts().sort_index()

# Display the counts
print(spf_range_counts)


# In[ ]:




