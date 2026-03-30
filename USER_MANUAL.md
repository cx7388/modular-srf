# Modular SRF Weight Elicitation Tool - User Manual

**Version:** 1.0  
**Date:** February 28, 2026  
**Author:** River Huang (river.huang@psi.ch)  
**Developed for:** Laboratory for Energy Systems Analysis (LEA), Paul Scherrer Institute (PSI)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Getting Started](#3-getting-started)
4. [User Interface Overview](#4-user-interface-overview)
5. [Step-by-Step Guide](#5-step-by-step-guide)
6. [SRF Methods Explained](#6-srf-methods-explained)
7. [Understanding the Results](#7-understanding-the-results)
8. [Import and Export Features](#8-import-and-export-features)
11. [Technical Details](#11-technical-details)
12. [References](#12-references)

---

## 1. Introduction

### 1.1 What is the Modular SRF Weight Elicitation Tool?

The Modular SRF Weight Elicitation Tool is a browser-based application designed to help decision-makers determine the importance weights of criteria in multi-criteria decision-making problems. It includes several predefined SRF methods together with a standalone Modular SRF framework that can combine procedural, informational, normative, and analytical components within one elicitation model.

### 1.2 When Should You Use This Tool?

Use this tool when you need to:
- Assign importance weights to multiple decision criteria
- Elicit preferences from decision-makers or experts
- Maintain transparency in the weight-setting process
- Analyze the robustness of criteria weights
- Handle uncertainty in preference information

### 1.3 Key Features

- **Interactive Deck-of-Cards Interface**: Arrange criteria cards visually to express preferences
- **Multiple SRF Configurations**: Choose from 7 predefined SRF methods plus the standalone Modular SRF framework
- **Real-Time Calculations**: Immediate weight computation and normalization
- **Robustness Analysis**: ASI diagnostics, sampling distributions, extreme-scenario heatmaps, and PCA visualizations
- **Import/Export**: Save and load configurations in JSON format
- **Export Results**: Download results to Excel (XLSX) format
- **No Installation Required**: Runs entirely in your web browser

---

## 2. Installation

### 2.1 System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Web Browser**: Modern browser (Chrome, Firefox, Safari, or Edge)
- **Memory**: At least 2 GB RAM recommended
- **Disk Space**: Approximately 500 MB for virtual environment and dependencies

### 2.2 Installation Steps

#### Step 1: Clone or Download the Repository

```bash
# If using git
git clone [repository-url]
cd srf-software

# Or download and extract the ZIP file
```

#### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- NumPy and Pandas (data processing)
- Scikit-learn (statistical analysis)
- PuLP and HiGHS (optimization solvers)
- Additional supporting libraries

#### Step 4: Verify Installation

After installation, you should see the following packages installed:
- Flask 3.1.0
- numpy 2.2.5
- pandas 2.2.3
- scikit-learn 1.5.2
- pulp 2.9.0
- highspy 1.8.1

---

## 3. Getting Started

### 3.1 Launching the Application

1. Open a terminal/command prompt
2. Navigate to the project directory
3. Activate the virtual environment (if not already activated)
4. Start the Flask server:

```bash
python -m flask --app simos_method run --port 8000
```

5. Open your web browser and navigate to:
```
http://localhost:8000
```

### 3.2 First Look

Upon opening the application, you'll see the **Home page** with:
- Navigation menu (Home, Elicitation Tool)
- Brief description of the tool
- Three key feature cards
- Citation information

Click on **"Open Elicitation Tool"** or use the navigation menu to access the main application.

### 3.3 Quick Start Example

Let's perform a simple weight elicitation:

1. On the Elicitation Tool page, you'll see **criterion cards** and **blank cards** on the right side
2. Drag criterion cards to the drop zone and arrange them from left (most important) to right (least important)
3. Insert blank cards between criteria to indicate preference intensity
4. Set the **z-value** (ratio between most and least important criteria)
5. If you are using a variability-oriented method, set the **sampling count** if needed (default: 200)
6. Click **"Calculate"** to compute the weights
7. Review the results displayed below

---

## 4. User Interface Overview

### 4.1 Main Components

#### Navigation Bar
- **PSI Logo**: Link to Paul Scherrer Institute
- **Home**: Return to the home page
- **Elicitation Tool**: Access the main application

#### Method Selection Dropdown
Select from 7 predefined SRF methods and the standalone Modular SRF framework:
- Simos-Roy-Figueira (SRF)
- SRF-II
- Robust SRF
- Assessment Through Prioritization (WAP)
- Imprecise SRF
- Belief-degree Imprecise SRF
- Hesitant Fuzzy Linguistic (HFL)-SRF
- Modular SRF

#### Method Instructions Panel
- Displays the full name and reference for the selected method
- Collapsible user guidelines section
- Links to academic papers (DOI links)

#### Card Stacks (Right Side)
- **Criterion Cards** (blue): Represent your decision criteria
- **Blank Cards** (white): Used to express preference intensity

#### Drop Zone (Center)
- Main working area where you arrange cards
- Horizontal arrow indicating "More important -> Less important"
- Drag and drop interface
- Column insertion button (+) to add blank columns

#### Control Buttons
- **Import**: Load a saved configuration
- **Export**: Save current configuration
- **Clear**: Remove all cards from the drop zone

#### Input Parameters
- **z-value**: Global ratio between most and least important groups
- **e0-value** (SRF-II only): Blank cards to hypothetical zero criterion
- **w-value**: Decimal precision for final weights
- **Sampling count** (variability-oriented methods): Number of feasible samples used for the sampling distribution and PCA views; default 200

#### Action Buttons
- **Calculate**: Compute the criteria weights
- **Export to XLSX**: Download results to Excel

#### Results Section
- Weight tables
- ASI (Average Stability Index) values
- Distribution plots (box plots)
- Extreme scenario heatmap
- PCA (Principal Component Analysis) visualizations

---

## 5. Step-by-Step Guide

### 5.1 Defining Your Criteria

#### Step 1: Add Criteria
1. Click on a **criterion card** in the criterion stack
2. Edit the text to name your criterion (e.g., "Cost", "Quality", "Safety")
3. Press Enter or click outside to confirm
4. Drag the card to the drop zone

#### Step 2: Add More Criteria
- Repeat the process for all your criteria
- You can add as many criteria as needed (typically 3-20)
- You can generate more cards by clicking existing ones in the stack

### 5.2 Ranking Criteria

#### Basic Ranking
1. Arrange cards from **left to right** in order of importance
   - Leftmost = Most important
   - Rightmost = Least important

#### Equal Importance (Ex Aequo)
- Place multiple cards in the same column to indicate equal importance
- Drag one card on top of another to create a column with multiple cards

#### Inserting Columns
- Click the **column insertion button** (+) to add a new blank column
- This helps separate cards more clearly

### 5.3 Expressing Preference Intensity with Blank Cards

Blank cards encode the **intensity of preference** between successive groups:

- **0 blank cards** between groups = 1 unit of difference
- **1 blank card** between groups = 2 units of difference  
- **2 blank cards** between groups = 3 units of difference
- **n blank cards** between groups = (n+1) units of difference

#### Example:
```
[Criterion A] [blank] [blank] [Criterion B] [Criterion C]
```
This means:
- A is 3 units more important than B (2 blank cards)
- B is 1 unit more important than C (0 blank cards)

### 5.4 Setting Parameters

#### Z-Value (Global Ratio)
- **Definition**: The ratio between the weight of the most important group and the least important group
- **Default**: 6.5
- **Range**: 1.5 to 1000
- **Example**: z=10 means the most important criterion is 10 times more important than the least important

#### E0-Value (SRF-II only)
- **Definition**: Number of blank cards between the least important criterion and a hypothetical "zero criterion"
- **Default**: 4
- **Range**: 0 to 999
- **Purpose**: Provides an alternative to the z-value

#### W-Value (Precision)
- **Definition**: Number of decimal places for final weights
- **Default**: 1
- **Range**: 0 to 2
- **Example**: w=2 gives weights like 0.23, w=1 gives 0.2, w=0 gives 0

#### Sampling Count (Variability-Oriented Methods)
- **Definition**: Number of feasible solutions used to build the sampling distribution and PCA views
- **Default**: 200
- **Range**: 1 to 20,000
- **Note**: For continuous SRF models, the sampler targets the uniform distribution over the feasible region

### 5.5 Calculating Weights

1. Verify your card arrangement is complete
2. Check that all criteria are included
3. Confirm your parameter values
4. Click the **"Calculate"** button
5. Wait for the computation (usually instant to a few seconds)
6. Review the results in the Results section below

---

## 6. SRF Methods Explained

### 6.1 Simos-Roy-Figueira (SRF) - The Original Method

**When to Use**: Standard scenarios with precise preference information

**How It Works**:
1. Rank criteria using cards
2. Use blank cards to express intensity
3. Provide global z-value
4. Model computes normalized weights

**Key Reference**: Figueira & Roy (2002), DOI: 10.1016/S0377-2217(01)00370-8

**Advantages**:
- Simple and intuitive
- Well-established method
- Transparent process

### 6.2 SRF-II - Zero-Criterion Approach

**When to Use**: When it's easier to think about absolute importance rather than relative ratios

**How It Works**:
1. Same ranking and blank cards as SRF
2. Instead of z, specify e0 (distance to "zero importance")
3. Weights derived from this absolute reference

**Key Reference**: Abastante et al. (2022), DOI: 10.1007/s12351-020-00611-4

**Advantages**:
- More intuitive for some users
- Avoids thinking about ratios
- Natural reference point

### 6.3 Robust SRF

**When to Use**: When you want to explore the space of compatible weights

**How It Works**:
1. Same input as SRF
2. Explores entire feasible weight region
3. Provides representative weights plus robustness analysis

**Key Reference**: Siskos & Tsotsolas (2015), DOI: 10.1016/j.ejor.2015.04.037

**Advantages**:
- Shows variability in compatible weights
- ASI values for robustness assessment
- Sampling distributions and PCA views
- Extreme-scenario heatmap for criterion-wise minima and maxima

**Output Includes**:
- Normalized weights
- ASI (Average Stability Index) value based on extreme scenarios
- Box plots showing sampled feasible weights
- Extreme scenario heatmap
- PCA plots (when applicable)

### 6.4 Assessment Through Prioritization (WAP)

**When to Use**: When you can provide local ratios between successive ranks

**How It Works**:
1. Rank criteria (no blank cards needed)
2. For each pair of successive ranks, provide ratio interval [z_min, z_max]
3. Model combines all local constraints

**Key Reference**: Tsotsolas et al. (2019), DOI: 10.1007/s12351-016-0280-7

**Advantages**:
- Finer control over preferences
- No need for blank cards
- Can express different intensities between different rank pairs

### 6.5 Imprecise SRF

**When to Use**: When you're uncertain about exact preference intensities

**How It Works**:
1. Rank criteria and identify relevant gaps
2. For each gap, provide interval [e_min, e_max] instead of exact number
3. Provide interval for z-value [z_min, z_max]
4. Model handles interval constraints

**Key Reference**: Corrente et al. (2017), DOI: 10.1016/j.omega.2016.11.008

**Advantages**:
- Accommodates uncertainty
- More realistic in practice
- Still produces actionable weights

### 6.6 Belief-Degree Imprecise SRF

**When to Use**: When you can assign probabilities to different scenarios

**How It Works**:
1. Rank criteria
2. For each gap and z-value, provide pairs (value, probability)
3. Probabilities must sum to 1
4. Model aggregates probabilistic inputs

**Key Reference**: Zhang & Liao (2023), DOI: 10.1080/01605682.2022.2035271

**Advantages**:
- Captures probabilistic beliefs
- More information than simple intervals
- Refined robustness diagnostics

### 6.7 Hesitant Fuzzy Linguistic (HFL)-SRF

**When to Use**: When you prefer linguistic terms over numerical values

**How It Works**:
1. Rank criteria
2. For successive ranks, choose linguistic intervals (e.g., "Low", "Medium", "High")
3. Uses predefined fuzzy sets
4. Model converts to numerical constraints

**Key Reference**: Wu et al. (2022), DOI: 10.1016/j.asoc.2022.108979

**Advantages**:
- Natural for non-technical users
- No need to think numerically
- Handles linguistic uncertainty

**Scales**:
- Rank-gap terms: 1-5 scale
- Global contrast terms: 1-10 scale

### 6.8 Modular SRF

**When to Use**: When you want to build a tailored elicitation model by combining SRF components instead of selecting one predefined method

**How It Works**:
1. Use the modular questionnaire to configure a standalone SRF framework
2. Combine procedural, informational, normative, and analytical components
3. The application assembles the corresponding optimization model
4. Compute weights and diagnostics for that custom configuration

**Key Reference**: Huang et al. (2026), DOI: 10.1016/j.eswa.2026.131315

**Advantages**:
- Standalone modular formulation
- Can combine components that also appear in predefined SRF methods
- Tailored to specific decision context

---

## 7. Understanding the Results

### 7.1 Weight Tables

After clicking "Calculate", the tool displays the computed criterion weights. The exact columns depend on the selected method. Crisp methods show a single weight vector, while variability-oriented methods may also show center, minimum, and maximum weights.

**Interpretation**:
- Weights are normalized (sum to 1.0 or 100%)
- Higher weight = more important criterion
- Can be used directly in MCDA methods (e.g., ELECTRE, PROMETHEE)

### 7.2 ASI Values (Robust Methods)

**ASI (Average Stability Index)**: A single summary value describing how stable the compatible weight solutions are across the feasible region.

For variability-oriented runs, the displayed ASI is computed from the **extreme scenario matrix** rather than from the sampled cloud. In other words, it summarizes how much criterion weights change across the criterion-wise minimum and maximum feasible solutions.

**Interpretation**:
- Values between 0 and 1
- Higher ASI = more robust/stable result
- ASI close to 1 = very stable compatible weights
- Lower ASI = greater variability in the feasible weight space

### 7.3 Distribution Plots (Box Plots)

For variability-oriented methods, the sampling distribution figure shows:
- **Box**: Interquartile range (IQR) - middle 50% of sampled weights
- **Line in Box**: Median sampled weight
- **Whiskers**: Sampled range for that criterion
- **Overlay line**: The displayed SRF weight vector from the results table

**Interpretation**:
- Wide box = high variability/uncertainty
- Narrow box = stable/robust weight
- Compare widths to assess relative robustness

**How the samples are generated**:
- In continuous SRF models, the tool uses hit-and-run sampling targeting a **uniform distribution** over the feasible region
- If a run includes discrete logical constraints, the tool falls back to feasible-solution exploration rather than continuous polytope sampling

### 7.4 Extreme Scenario Heatmap

The extreme scenario heatmap summarizes the boundary solutions used for robustness interpretation.

**Elements**:
- **Rows**: Extreme scenarios obtained by minimizing or maximizing one criterion at a time
- **Columns**: Criteria
- **Cell color/value**: Weight of a criterion in that extreme scenario

**Interpretation**:
- Dark/light shifts show which criteria move most across the feasible region
- Similar rows indicate that several extrema lead to comparable weight patterns
- Strong contrasts highlight criteria that are especially sensitive to the imposed preference constraints

### 7.5 PCA Plots

**PCA (Principal Component Analysis)**: 2D visualization of weight space

**Elements**:
- **Points**: Sampled feasible weight vectors
- **Clusters**: Groups of similar weights
- **Distribution**: Spread indicates robustness

For continuous SRF models, the PCA cloud is built from the same uniformly sampled hit-and-run solutions used in the boxplot view.

**Interpretation**:
- Tight cluster = stable recommendations
- Scattered points = high variability
- Multiple clusters = potentially different preference scenarios

### 7.6 Statistical Summaries

Depending on the method, you may see additional statistics:
- **Mean weights**: Average across samples
- **Standard deviation**: Measure of variability
- **Min/Max values**: Range of compatible weights
- **Confidence intervals**: Statistical bounds on weights

---

## 8. Import and Export Features

### 8.1 Exporting Your Configuration

**Purpose**: Save your current card arrangement and parameters for later use

**Steps**:
1. Arrange your cards and set parameters
2. Click the **"Export"** button
3. A JSON file will be downloaded to your computer
4. Default filename: `srf_configuration_[timestamp].json`

**What's Saved**:
- Card positions and labels
- Blank card positions
- Selected method
- Parameter values (z, e0, w)
- Additional method-specific settings

### 8.2 Importing a Configuration

**Purpose**: Load a previously saved configuration

**Steps**:
1. Click the **"Import"** button
2. Select a JSON file from your computer
3. The interface will automatically populate with the saved configuration
4. Review and modify if needed
5. Click "Calculate" to compute weights

**Use Cases**:
- Resuming previous work
- Comparing different scenarios
- Sharing configurations with colleagues
- Version control of decision models

### 8.3 Exporting Results to Excel

**Purpose**: Download calculation results for further analysis or reporting

**Steps**:
1. After calculating weights, click **"Export to XLSX"**
2. An Excel file will be downloaded
3. Default filename: `srf_results_[timestamp].xlsx`

**What's Included**:
- Sheet 1: Criteria Weights
- Sheet 2: Sampling Results (if variability analysis is active)
- Sheet 3: Extreme Scenarios (if variability analysis is active)

The additional variability sheets are omitted when no corresponding data is available.

**File Format**: Microsoft Excel (.xlsx) format, compatible with Excel, LibreOffice, Google Sheets

---

## 11. Technical Details

### 11.1 System Architecture

**Frontend**:
- HTML5, CSS3, JavaScript (ES6+)
- Drag-and-drop API
- Plotly.js for visualizations
- XLSX.js for Excel export

**Backend**:
- Flask 3.1.0 (Python web framework)
- RESTful API architecture
- JSON data exchange

**Computation**:
- NumPy for numerical operations
- Pandas for data manipulation
- Scikit-learn for PCA and statistical analysis
- PuLP for optimization modeling
- HiGHS as optimization solver

**Core project files**:
- `simos_method/__init__.py`: Flask routes, input normalization, deck preprocessing, and response formatting
- `simos_method/static/python/srf_methods.py`: SRF structure resolution, optimization, sampling, and inconsistency analysis
- `simos_method/static/python/utils.py`: rounding, ASI, and PCA export helpers
- `simos_method/static/python/freeopt.py`: compatibility layer that maps the used subset of the `gurobipy` API to free solvers
- `simos_method/static/js/uiUtils.js`: method-specific inputs and modular questionnaire behavior
- `simos_method/static/js/backend.js`: browser-side payload serialization and POST `/calculate`
- `simos_method/static/js/results.js`: results table plus sampling distribution, extreme-scenario heatmap, and PCA rendering

**Request flow**:
1. The browser collects the card arrangement and method-specific inputs.
2. `backend.js` sends them to the Flask `/calculate` endpoint.
3. `simos_method/__init__.py` validates and reshapes the payload into the format expected by the solver layer.
4. `srf_methods.py` runs the requested SRF workflow and writes any plot-support files.
5. The browser renders the returned table and loads the variability/progress JSON files when needed.

### 11.2 Optimization Models

The tool formulates **linear programming (LP)** problems:

**Variables**: Criteria weights w1, w2, ..., wn

**Constraints**:
- Normalization: Î£wáµ¢ = 1
- Non-negativity: wi >= 0
- Rank-order: wi >= wj if i is more important than j
- Intensity: Based on blank cards
- Ratio: Based on z-value
- Additional: Method-specific constraints

**Objective**: Varies by method (e.g., maximize sum, minimize deviation)

### 11.3 Solver Configuration

**Primary Solver**: HiGHS (High-Performance LP solver)
- Open-source
- Fast and reliable
- Handles large problems

**Fallback Solver**: PuLP's default solvers
- CBC (COIN-OR Branch and Cut)
- Used if HiGHS unavailable

**Solver Warmup**: 
- First call to solver can be slow
- Tool "warms up" solver at startup
- Subsequent calculations are fast

**Useful environment variables**:
- `FREEOPT_SOLVER=auto|highs|cbc`: choose the preferred free solver backend
- `FREEOPT_THREADS=<int>`: limit solver threads
- `FREEOPT_TIME_LIMIT_SEC=<int>`: apply a solver time limit
- `SRF_SKIP_SOLVER_WARMUP=1`: disable the startup warmup solve

### 11.4 Data Storage

**Configuration Files**:
- Location: `simos_method/static/data/`
- `simos_instructions.json`: Method descriptions
- `srf_samples.json`: Cached sampling results
- `srf_extreme_scenarios.json`: Cached labeled extreme scenarios
- `pca_output.json`: Cached PCA results
- `srf_export_payload.json`: Cached XLSX export payload for variability details
- `calculation_progress.json`: Cached progress payload for the active calculation

**File Formats**:
- Input/Output: JSON
- Results Export: XLSX (Excel)

**Browser Storage**:
- No persistent storage in browser
- All data lost on page refresh unless exported

**Runtime behavior**:
- `srf_samples.json`, `srf_extreme_scenarios.json`, `pca_output.json`, `srf_export_payload.json`, and `calculation_progress.json` are recreated during each calculation
- The backend writes placeholder JSON before solving so the frontend does not hit temporary 404 errors while plots are loading

### 11.5 Performance Considerations

**Typical Performance**:
- 5-10 criteria: Instant (<1 second)
- 20 criteria: 1-2 seconds
- 50 criteria: 5-10 seconds
- 100 criteria: 30-60 seconds (not recommended)

**Robust Methods**:
- Use a default sampling count of 200, configurable in the UI
- For continuous models, sampling targets the uniform distribution over the feasible region via hit-and-run
- Mixed-integer logical variants fall back to feasible-solution exploration
- Adds 5-30 seconds depending on criteria count
- Variability outputs are written to JSON files for frontend loading and XLSX export

**Optimization**:
- Warm start solver at launch
- Cache results when possible
- Use efficient LP formulations

### 11.6 Browser Compatibility

**Fully Supported**:
- Google Chrome 90+
- Mozilla Firefox 88+
- Microsoft Edge 90+
- Safari 14+

**Partial Support** (may have minor issues):
- Older browsers (consider updating)
- Mobile browsers (interface designed for desktop)

**Requirements**:
- JavaScript enabled
- Cookies enabled
- Modern ES6 support
- Canvas support (for visualizations)

### 11.7 Security and Privacy

**Data Privacy**:
- All computation happens locally on your machine
- No data sent to external servers
- No tracking or analytics

**Security**:
- Local deployment only
- No authentication required
- No database (stateless)
- File uploads validated (JSON only)

**Recommendations**:
- Do not expose to public internet
- Use on trusted networks only
- Keep virtual environment isolated

---

## 12. References

### 12.1 Primary Reference

Please cite this tool using:

**APA Format**:
```
Huang, R., Kadzinski, M., Figueira, J. R., Corrente, S., Siskos, E., & Burgherr, P. (2026). 
A Modular Simos-Roy-Figueira Framework for Tailored Weight Elicitation in Multi-Criteria 
Decision Aiding. Expert Systems with Applications, 311, 131315. 
https://doi.org/10.1016/j.eswa.2026.131315
```

**BibTeX Format**:
```bibtex
@article{huang2026modular,
  title={A Modular Simos-Roy-Figueira Framework for Tailored Weight Elicitation in Multi-Criteria Decision Aiding},
  author={Huang, River and Kadzi{\'n}ski, Mi{\l}osz and Figueira, Jos{\'e} Rui and Corrente, Salvatore and Siskos, Eleftherios and Burgherr, Peter},
  journal={Expert Systems with Applications},
  volume={311},
  pages={131315},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2026.131315}
}
```

### 12.2 Method-Specific References

**SRF (Original)**:
- Figueira, J., & Roy, B. (2002). Determining the weights of criteria in the ELECTRE type methods with a revised Simos' procedure. *European Journal of Operational Research, 139*(2), 317-326. https://doi.org/10.1016/S0377-2217(01)00370-8

**SRF-II**:
- Abastante, F., Corrente, S., Greco, S., Lami, I. M., & Mecca, B. (2022). The introduction of the SRF-II method to compare hypothesis of adaptive reuse for an iconic historical building. *Operational Research, 22*(3), 2397-2436. https://doi.org/10.1007/s12351-020-00611-4

**Robust SRF**:
- Siskos, E., & Tsotsolas, N. (2015). Elicitation of criteria importance weights through the Simos method: A robustness concern. *European Journal of Operational Research, 246*(2), 543-553. https://doi.org/10.1016/j.ejor.2015.04.037

**WAP**:
- Tsotsolas, N., Spyridakos, A., Siskos, E., & Salmon, I. (2019). Criteria weights assessment through prioritizations (WAP) using linear programming techniques and visualizations. *Operational Research, 19*(1), 135-150. https://doi.org/10.1007/s12351-016-0280-7

**Imprecise SRF**:
- Corrente, S., Figueira, J. R., Greco, S., & Slowinski, R. (2017). A robust ranking method extending ELECTRE III to hierarchy of interacting criteria, imprecise weights and stochastic analysis. *Omega, 73*, 1-17. https://doi.org/10.1016/j.omega.2016.11.008

**Belief-Degree Imprecise SRF**:
- Zhang, Z., & Liao, H. (2023). An evidential reasoning-based stochastic multi-attribute acceptability analysis method for uncertain and heterogeneous multi-attribute reverse auction. *Journal of the Operational Research Society, 74*(1), 239-257. https://doi.org/10.1080/01605682.2022.2035271

**HFL-SRF**:
- Wu, H., Ren, P., & Xu, Z. (2022). Promoting the physician-patient consensus with a hesitant fuzzy linguistic consensus method based on betweenness relation. *Applied Soft Computing, 124*, 108979. https://doi.org/10.1016/j.asoc.2022.108979

### 12.4 Contact and Support

**Developer**: River Huang  
**Email**: river.huang@psi.ch  
**Institution**: Paul Scherrer Institute (PSI)  
**Laboratory**: Laboratory for Energy Systems Analysis (LEA)  
**Website**: https://www.psi.ch/en/lea

**For Questions**:
- Technical issues: Contact developer via email
- Method questions: Refer to academic papers
- Feature requests: Contact developer via email

**Acknowledgments**:
This tool was developed as part of research conducted at the Laboratory for Energy Systems Analysis (LEA) at the Paul Scherrer Institute (PSI), Switzerland.

---

## Appendix A: Glossary

**ASI (Average Stability Index)**: A scalar indicator describing how stable the compatible weight solutions are across the feasible region; in the interface it is reported from the extreme-scenario matrix.

**Blank Card**: A white card used to express preference intensity between criteria.

**Criterion**: A factor or dimension used to evaluate alternatives in decision-making.

**Drop Zone**: The central area where you arrange criterion and blank cards.

**E0 (e-zero)**: The number of blank cards between the least important criterion and a hypothetical zero-importance criterion.

**Ex Aequo**: Latin term meaning "equal"; criteria with equal importance placed in the same rank/column.

**Feasible Weight Space**: The set of all weight vectors that satisfy the preference constraints.

**HFL (Hesitant Fuzzy Linguistic)**: An approach using linguistic terms with fuzzy uncertainty.

**LP (Linear Programming)**: Mathematical optimization method for finding the best outcome in a linear model.

**Normalized Weights**: Weights scaled to sum to exactly 1.0 (or 100%).

**PCA (Principal Component Analysis)**: A statistical technique for visualizing high-dimensional data in 2D or 3D.

**Preference Intensity**: The strength of preference for one criterion over another.

**Rank-Order**: The ordering of criteria from most to least important.

**Robust Analysis**: Examination of how results vary across the feasible preference space.

**SRF (Simos-Roy-Figueira)**: The revised Simos' method and its framework.

**W-Value**: The number of decimal places for final weight precision.

**Z-Value**: The ratio between the weight of the most important and least important criterion groups.

---

## Appendix C: Quick Reference Card

### Essential Steps

1. **Launch**: `python -m flask --app simos_method run --port 8000`
2. **Navigate**: `http://localhost:8000`
3. **Select Method**: Choose from dropdown
4. **Add Criteria**: Edit and drag criterion cards
5. **Rank**: Arrange left (important) to right (less important)
6. **Intensity**: Insert blank cards between groups
7. **Parameters**: Set z, e0, w, and sampling count when variability analysis is active
8. **Calculate**: Click calculate button
9. **Review**: Check weights and diagnostics
10. **Export**: Save configuration or results

### Keyboard Shortcuts

- **Enter**: Confirm criterion edit
- **Escape**: Cancel criterion edit
- **Delete**: Remove focused card (if implemented)
- **Ctrl+S**: (Browser) Save page
- **F5**: Refresh page
- **F12**: Open browser developer tools

### Parameter Quick Guide

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| z-value | 6.5 | 1.5-1000 | Global ratio (most/least important) |
| e0-value | 4 | 0-999 | Distance to zero criterion (SRF-II) |
| w-value | 1 | 0-2 | Decimal precision of final weights |
| Sampling count | 200 | 1-20000 | Number of feasible samples used in variability figures |

### Common Patterns

**High Differentiation**:
- Use many blank cards
- High z-value (8-20)

**Low Differentiation**:
- Few blank cards
- Low z-value (2-4)

**Equal Groups**:
- Multiple cards in same column
- No blank cards between them

**High Uncertainty**:
- Use Robust/Imprecise methods
- Review ASI, distributions, and extreme scenarios

---

## Appendix D: Frequently Asked Questions (FAQ)

**Q1: How many criteria should I use?**  
A: Typically 5-15 criteria is manageable. Fewer than 5 may be too simple; more than 20 becomes cognitively challenging.

**Q3: What if two criteria are truly equal in importance?**  
A: Place them in the same column (ex aequo). They will receive exactly equal weights.

**Q4: Should I use SRF or Robust SRF?**  
A: Use SRF for a single weight vector. Use Robust SRF when you want variability diagnostics such as sampling distributions, extreme scenarios, ASI, and PCA.

**Q5: What z-value should I choose?**  
A: Common values: 5-10. Higher values (15-20) indicate strong differentiation. Lower (2-4) indicate mild differences.

**Q6: Can I use this tool for multiple decision-makers?**  
A: Yes, but each decision-maker should elicit separately, then aggregate results (e.g., average weights or compare differences).

**Q8: What if I'm uncertain about rankings?**  
A: Use robust methods to explore multiple rankings, or use imprecise/belief-degree methods to model uncertainty.

**Q9: Can I save my work and continue later?**  
A: Yes, use the Export button to save configuration to JSON, then Import later to resume.

**Q10: Why are my ASI values all very high or very low?**  
A: Very high ASI (close to 1) indicates that the extreme scenarios remain similar across criteria. Lower ASI means the compatible weights vary more strongly across those extreme solutions.

**Q11: Can I use fractional blank cards?**  
A: No, only whole blank cards. Use intervals (Imprecise SRF) for fractional differentiation.

**Q12: What's the difference between SRF and SRF-II?**  
A: SRF uses a ratio (z) between most and least important. SRF-II uses distance to hypothetical zero (e0). Choose based on intuition.

**Q13: How accurate are the weights?**  
A: Weights precisely reflect your input preferences. Validity depends on quality of your preference elicitation.

**Q14: Can I use this for real-world decisions?**  
A: Yes, but always validate results with stakeholders and consider sensitivity analysis.

**Q15: Is there a mobile version?**  
A: The interface is designed for desktop/laptop. Mobile browsers may work but with reduced usability.

---

## End of User Manual

**Document Version**: 1.0  
**Last Updated**: February 28, 2026  
**Tool Version**: Based on srf-software repository  

For the latest version of this manual and the software, please contact the developer or visit the official repository.

**Thank you for using the Modular SRF Weight Elicitation Tool!**


