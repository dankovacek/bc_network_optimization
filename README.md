# BC Streamflow Monitoring Optimization

A common use of streamflow data is to estimate volume and timing of streamflow in unobserved locations.  A number of different model approaches are used, such as process-based, conceptual, empirical, and machine learning.  The accuracy of these models is dependent on the quality and quantity of the observed data.  Any hydrological model approch requires long-term observations in the order of decades, so for today's studies we have only a limited number of locations with sufficient data, where the locations themselves were established in previous decades from priorities.  The result is that the locations of streamflow monitoring stations are not necessarily optimal for the needs of today or the future.  

In this study we look at streamflow monitoring network optimization through the lens of information theory.  We use the concept of surprise to quantify the information content of a streamflow observation, and we use the concept of expected surprise reduction to quantify the information content of a potential monitoring location.  We use a greedy network expansion algorithm to find the location(s) that maximize the expected total reduction in surprise for the set of all ungauged locations.  In other words, we find the ungauged location(s) that provide the most (expected) reduction in uncertainty for the ungauged space.

The project is implemented in Python, and the general methodology is as follows:

* **setup**: set up the project, including downloading data and setting up the Python environment
* **data preproccessing**: prepare the data for analysis
* **analysis**: perform the analysis
* **results**: summarize the results

## Getting Started

Clone this repo and collect the required input data.  The project is implemented in Python, and the required packages are listed in `requirements.txt`.

### Prerequisites

To perform a network optimization, the following is required:

* **British Columbia Ungauged Basins**: monitoring network optimization requires characterization of a decision space of potential monitoring locations.  We use the British Columbia Ungauged Basins (BCUB) dataset (citation?) to represent the decision space for BC hydrometric monitoring.  The BCUB is a set of attributes describing the terrain, soil, land cover, and climate for a very large set of ungauged locations.  The BCUB dataset is available from the [Open Science Foundation data repository](https://osf.io/6p7ae/) and should be retrieved and saved to disk, or an alternate set of basins and attributes can be used to apply the method to a different region of interest.  
* **Streamflow monitoring network**: we use the HYSETS dataset (Arsenault, 2019) to represent the active and discontinued streamflow monitoring network in BC and trans-boundary basins.  This dataset is used to develop a model to map the information content of basin attributes to the expected divergence of the streamflow distribution from the prior distribution.  The HYSETS dataset is available from the [Open Science Foundation data repository](https://osf.io/rpc3w/) and should be retrieved and saved to disk.

For information on the development of BCUB, see the [BCUB repository](https://github.com/dankovacek/bcub).  For a more detailed walkthrough of how to process attributes for a new region of interest, see the [BCUB demo repository](https://github.com/dankovacek/bcub_demo).

## Study Region

![Pour points in the study region overlaid by active and discontinued monitoring locations.](img/main_fig_updated.png)

In the figure above, the British Columbia adminstrative boundary (red dashed line) represents the decision space within which streamflow monitoring network decisions can be made.  Also shown are active (green dots) and discontinued (orange dots) streamflow monitoring locations.  Light grey dots represent the approximately 1.2 million pour points in the BCUB ungauged basins for which attribute sets have been generated. 

A buffer extends the study region bounds extend beyond the BC administrative border to incorporate trans-boundary basins and other adjacent regions (by basin) that are close in space to the border that may contain information relevant to the decision space.  Additionally, the buffer mitigates edge effects of the decision space boundary on the optimization algorithm.  

## Data Preprocessing

The data preprocessing steps includes adding columns to the ungauged basin dataset to describe baseline distances and expected surprise reduction.  

### Basin Polygon Validation

HYSETS is missing lots of basins, they can be infilled by querying the USGS NWIS API (or by searching the WSC updated polygons set) for either an official polygon (grade A), or coordinates describing the station location **and** basin centroid(grade B), or :

| Source / Criteria | 1 (>=95% TP, <5% FP/FN) | 2 (>=95% TP, <10% FP/FN) | 3 (>50% TP, <50% FP/FN) |
|-------------------|-----------------------------------|------------------------------------|-------------------------|
| A. Polygon  | High accuracy for polygons. | Good accuracy but moderate precision for polygons. | Acceptable accuracy, lower precision for polygons. |
| B. Pour Point + Centroid + Area  | Pour point and basin centroid within 500m, and area within 5%. | Pour point, basin centroid within 500m, basin area within 10%. | Pour point and basin centroid within 500m, area within 25%. |
| C. Est. Pour Point | High accuracy and precision for estimated pour points. | Good accuracy but moderate precision for estimated pour points. | Acceptable accuracy, lower precision for estimated pour points. |

1. True Positive (TP): the validation polygon overlaps the official polygon, 
2. False Positive (FP): the validation polygon does not overlap the official polygon,
3. False Negative (FN): the official polygon does not overlap the validation polygon.

Accuracy includes TP, FP, and FPN, and is given by:

$$Accuracy = \frac{TP}{TP + FP + FN}$$

### Basin Attributes

Add the climate attributes by clipping each of the daymet rasters with the monitored basin polygon set.  This step is done in the `extend_hysets.py` script.  The ten daymet rasters were derived in the BCUB dataset, and they are provided in the `input_data` folder, and they correspond to the six daymet parameters (precip, max. temp., min. temp., vapour pressure, shortwave radiation, snow water equivalent), and four computed precip indices (high and low precipitation duration and frequency).

### Baseline Distances

Calculate the baseline distances from each ungauged location to the nearest active monitoring location.  This is used as a basis of comparison for the network optimization.  The "baseline distance" is mapped to the expected surprise reduction using the model developed in the analysis step.  The baseline distance represents the expected divergence between the streamflow distribution at the ungauged location and the prior distribution, represented by streamflow mapped from an active monitoring station "most similar" to each ungauged location.  The baseline distance for each ungauged location is calculated using the `calc_baseline_distances.py` script.

### Daymet Climate Attributes

The Daymet climate attributes are calculated for each basin in the HYSTS dataset.  This is done in the preprocessing step since these attributes are not included in the main attributes table published in the HYSETS dataset.  

## Models

The goal of this project is to determine if recommendations for streamflow monitoring can be supported with basin attributes.  This approach requires a model to map basin attributes to the expected divergence of the streamflow distribution from the prior distribution.  

We propose several models to address slightly different questions, since streamflow observations are used across broad domains of application.

All models are developed using pairwise comparisons between streamflow monitoring locations, and different models are suited to different applications.  The first model is more suited to evaluating long-term prediction where the long-term distribution of streamflow is of interest, such as water balance.  The second model is more suited to evaluating shorter-term prediction applications where the timing of streamflow is more critical.

### Model 1: Long-term streamflow distribution comparison

The first model describes the information loss due to simulating flow at an unobserved location based on observations at another location.  The "information loss" is the Kullback-Leibler divergence ($D_{KL}$) between the prior distribution ($P$, simulated) and the posterior distribution ($Q$, observed).  The comparison of long-term distributions reflects applications that are less concerned with timing and magnitude of individual events and more appropriate for long-term estimates, such as water balance.  

The $D_{KL}$ is calculated for each pair of stations in the HYSETS dataset in both directions since $D_{KL}(P||Q) \neq D_{KL}(Q||P)$.  This is reflected in the fact that using one basin for estimating   The $D_{KL}$ is then mapped to the basin attributes for each pair of stations to develop a model to predict the $D_{KL}$ for a given set of basin attributes.  The model is then used to predict the $D_{KL}$ for each ungauged location, and the $D_{KL}$ is mapped to the expected surprise reduction.

We use pairs of locations such that we can compare the recorded observation against a simulation.  For all pairwise comparisons, we use the terms proxy to represent the observed location, and target to represent where we want to predict flow.  Using the HYSETS dataset, we first find all pairs of stations meeting the following criteria:

1. The distance between basin centroids is less than 500 km.
2. The concurrent period of record is at least 20 years.

Since climate inputs are a major driver of streamflow, and the timing and magnitude of precipitation and temperature are more likely to be correlated over shorter distances, we limit the analysis to basins within 500 km of each other, though this is considered a very low (inclusive) bar.  We also limit the analysis to basins with at least 20 years of concurrent record to ensure that the $D_{KL}$ is based on a sufficiently large sample.  The minimum concurrent record length represents a tradeoff between the sample size that the $D_{KL}$ is based on and the number of pairs of stations that meet the criteria.  Reducing the threshold would incorporate a larger sample of basins but decrease the sample size for computing $D_{KL}$.  We test the senstitivity of the results to this threshold in the analysis step.

The $D_{KL}$ is interpreted as the additional information required to update the prior distribution to the posterior distribution.  The $D_{KL}$ is calculated as:

$$D_{KL}(P||Q) = \sum_{i=1}^n P(i) \log \frac{P(i)}{Q(i)}$$

where $P$ is the prior distribution and $Q$ is the posterior distribution.  In this case, $P$ is the daily flow distribution at the upstream station and $Q$ is the daily flow distribution at the downstream station.

Since $D_{KL}(P||Q) \neq D_{KL}(Q||P)$, we compute the $D_{KL}$ in both directions since mapping one (observed) location as a model proxy to map daily flows to an ungauged location is not symmetrical because $D_{KL}$ is computed on discrete distributions.  The distributions are set by discretizing the flow series into 8 bit representation, which is a reasonable assumption given that streamflow is only indirectly measured and at best measurement (rating curve) uncertainty is between 5-10%.

The following piecewise function is used to map daily flows from the upstream station to the downstream station:

$$
Q_{target} = \begin{cases} 
      0 & Q_{proxy} < \frac{Q_{intercept}}{} \\
      \frac{A_{target}}{A_{proxy}}Q_{proxy} & Q_{upstream} > Q_{intercept}
   \end{cases}
$$

### Model 2: Entropy Rate

The second model computes the bitrate of the residual series of simulated daily streamflow at an ungauged location.  Simulation of daily flows is simply done by scaling flows from an observed location by the ratio of drainage area ($X_{sim} = \frac{A_{sim}}{A_{prox}} X_{prox}$) and the residual series is the difference between observed and simulated flow ($X_{residual} = X_{sim} - X_{obs}$).  The bitrate is the entropy rate of the residual series, which is the minimum number of bits required to encode the residual series.  If the (area ratio) model were perfect, the residuals would all be zero and the bitrate would be zero.  In other words, the proxy location and the ratio of drainage areas would contain complete information about the target (simulated) location.  This is not the case, however, and the bitrate is a measure of the information content of the residual series.  The compression ratio is the ratio of the bitrate to the entropy rate of the observed series, and is a measure of the information content of the residual series relative to the observed series.  

The bitrate is calculated as:


The bitrate is calculated for each pair of stations in the HYSETS dataset in both directions since the bitrate is not symmetrical.  The bitrate is then mapped to the basin attributes for each pair of stations to develop a model to predict the bitrate for a given set of basin attributes.  The model is then used to predict the bitrate for each ungauged location, and the bitrate is mapped to the expected surprise reduction.

Given a large sample of observed locations, the purpose of the study is to determine to what extent the bitrate can be predicted by basin attributes. In other words, is there enough information in attributes to be able to decide between potential (candidate) monitoring locations?  A good classifier should be able to compare the attributes of two locations and predict the bitrate with high certainty.  We use a classification model to predict the *expected* bitrate of classes (A, B, ...), and express the choice of a potential monitoring location as a net benefit to prediction at all other ungauged locations (within some distance). The benefit of adding a streamflow monitoring location at some location is expressed as the *total reduction in surprise* of the total space (weighted?).