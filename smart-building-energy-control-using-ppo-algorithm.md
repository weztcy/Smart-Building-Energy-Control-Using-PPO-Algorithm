# Portfolio Project Data — Smart Building Energy Control Using PPO Algorithm

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/smart_building_energy_control/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image12.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "smart-building-energy-control-using-ppo-algorithm",
  "code": "PRO-34 // ML-19",
  "title": "Smart Building Energy Control Using PPO Algorithm",
  "category": "machine-learning",
  "categoryLabel": "Reinforcement Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Smart-Building-Energy-Control-Using-PPO-Algorithm",
  "license": "MIT License",
  "banner": "/images/projects/smart_building_energy_control/banner.webp",
  "summary": "PPO reinforcement learning controller optimizing HVAC setpoints, cutting energy use 8.96% vs fixed thermostats and 99.2% of comfort degree-hours.",
  "projectDescription": "Reinforcement learning system using Proximal Policy Optimization inside a physics-based 1R1C Gymnasium building simulator. The PPO agent modulates continuous HVAC setpoints every 15 minutes to balance electricity demand and ASHRAE 55 thermal comfort across four multi-seed operating scenarios.",
  "architectureSummary": "1R1C Thermodynamic Gymnasium Simulator -> 8D Normalized State -> PPO Actor-Critic MLP -> Continuous HVAC Setpoint Control & Multi-Scenario Evaluation",
  "techStack": [
    "Python",
    "Stable-Baselines3",
    "PyTorch",
    "Gymnasium",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "PyYAML",
    "Pytest"
  ],
  "telemetryMatrix": {
    "projectType": "Reinforcement Learning HVAC Control",
    "targetPlatform": "Python Simulation & Streamlit UI",
    "primaryDomain": "Smart Building Energy Optimization",
    "architecturePipeline": "1R1C Env -> PPO MLP -> Multi-Seed Eval",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Continuous RL Control System",
    "learningParadigm": "Reinforcement Learning",
    "modelDomain": "Smart Building HVAC Control",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Continuous PPO Control",
      "1R1C Thermal Physics",
      "Multi-Scenario Benchmark"
    ]
  },
  "mlProblemContext": "Commercial HVAC systems consume over 40% of building electricity, yet rule-based schedules suffer severe morning discomfort from thermal inertia lag while static thermostats waste energy conditioning empty rooms. An adaptive agent is needed to anticipate heat loads and maintain ASHRAE 55 comfort efficiently.",
  "mlDatasetOverview": {
    "profileBadge": "1R1C Online Gymnasium Simulator",
    "source": "Custom SmartBuildingEnv with procedural WeatherGenerator and OccupancyGenerator",
    "modality": "Continuous Dynamical State-Action Simulation & Time-Series Logs",
    "scale": "100,000 training timesteps (149 7-day episodes @ 15-min steps; 672 steps/episode)",
    "inputShape": "Box(8,) normalized float32 observation vector",
    "dataFormat": "Gymnasium Float32 Tensors / CSV / JSON / NPZ",
    "description": "Online closed-loop simulation modeling a 120 m² commercial office zone over 7-day Monday-to-Sunday cycles. Training runs on the standard scenario (seed 42), followed by held-out 3-seed evaluation across standard, heatwave, high-occupancy, and weekend conditions.",
    "provenanceNote": "All thermodynamic parameters, weather waves, occupancy profiles, and ASHRAE 55 comfort bands are self-contained and reproducible via experiment_config.yaml."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "8D State & 1D Action Space",
    "spaceNote": "All eight physical, occupancy, and temporal state features are bounded within [-1.0, 1.0].",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Continuous HVAC Cooling Setpoint",
      "type": "Continuous Scalar Action (20.0 °C to 26.0 °C)",
      "distribution": "Action a_t in [-1.0, 1.0] mapped via T_set = 23.0 + 3.0 * a_t"
    },
    "inputSignals": [
      {
        "name": "Indoor Air Temperature (T_in)",
        "type": "Continuous Normalized Scalar",
        "importanceOrRole": "Primary thermal feedback state",
        "description": "Zone temperature integrated via 1R1C heat balance and scaled from [16, 30] °C into [-1, 1]."
      },
      {
        "name": "Outdoor Air Temperature (T_out)",
        "type": "Continuous Normalized Scalar",
        "importanceOrRole": "Envelope conduction & COP driver",
        "description": "Diurnal and synoptic external temperature scaled from [16, 40] °C, directly governing chiller COP."
      },
      {
        "name": "Zone Occupancy Count (N_occ)",
        "type": "Normalized Count Signal",
        "importanceOrRole": "Internal heat & comfort trigger",
        "description": "Simulated office headcount (0–25 occupants) driving sensible heat gains and occupied comfort bounds."
      },
      {
        "name": "Cyclical Time Encoding (sin/cos hour)",
        "type": "Periodic Temporal Pair",
        "importanceOrRole": "Look-ahead pre-cooling cue",
        "description": "Sine and cosine projections of 24-hour time enabling smooth anticipatory cooling before 08:00 arrivals."
      },
      {
        "name": "Workday & Work-Hour Indicators",
        "type": "Binary Regime Flags",
        "importanceOrRole": "Schedule context discriminator",
        "description": "Bipolar {-1.0, 1.0} indicators separating weekday 08:00–18:00 shifts from nights and weekends."
      },
      {
        "name": "Previous HVAC Setpoint (T_set_prev)",
        "type": "Continuous Autoregressive State",
        "importanceOrRole": "Control stability regularizer",
        "description": "Prior step setpoint normalized to [-1, 1], supporting smooth actuation under the delta-action penalty."
      }
    ],
    "dataCharacteristicsSummary": "Observation dynamics couple slow 1R1C building thermal mass (1.2e7 J/K capacitance) with sharp weekday arrival ramps and midday solar irradiance up to 950 W/m². Temperature-dependent chiller COP drops during hot afternoons, penalizing poorly timed cooling."
  },
  "mlLearningApproach": {
    "paradigm": "Model-Free On-Policy Reinforcement Learning (MDP Control)",
    "methodologyDescription": "Proximal Policy Optimization learns a stochastic Gaussian control policy over continuous HVAC setpoints using clipped surrogate advantages. Multi-objective step rewards combine normalized electric kW cost, quadratic comfort deviations, and squared action deltas to enforce smooth pre-cooling.",
    "justification": "PPO prevents destructive policy updates via probability-ratio clipping without needing large off-policy replay buffers. Its continuous action parameterization avoids discrete thermostat chatter and learns anticipatory pre-cooling strategies that rule-based schedules cannot express."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "1R1C Thermodynamic State Integration",
      "description": "Compute envelope conduction, internal occupant/equipment heat, and window solar gain every 15 minutes to update zone temperature with Gaussian process noise.",
      "badge": "Physics Step"
    },
    {
      "step": "02",
      "title": "Bounded State & Cyclical Time Scaling",
      "description": "Normalize indoor/outdoor temperatures, occupancy count, and previous setpoint into [-1, 1] and encode 24-hour time using sine and cosine transforms.",
      "badge": "Normalization"
    },
    {
      "step": "03",
      "title": "Affine Action-to-Setpoint Mapping",
      "description": "Clip policy action outputs to [-1.0, 1.0] and map linearly into physical HVAC cooling setpoints across the 20.0 °C to 26.0 °C operating range.",
      "badge": "Action Scaling"
    },
    {
      "step": "04",
      "title": "Multi-Objective Reward Shaping",
      "description": "Evaluate normalized electrical power, occupancy-weighted quadratic ASHRAE 55 violation penalty, and squared control-delta penalty at every simulation step.",
      "badge": "Reward Signal"
    }
  ],
  "mlDataPreparationNote": "Training uses seed 42 exclusively on the standard scenario, while all final benchmarks run deterministically across three independent held-out evaluation seeds (42, 101, 2024) with synchronized RNG streams.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Proximal Policy Optimization (PPO Actor-Critic MLP)",
    "architectureSummary": "Separate two-layer Multilayer Perceptron actor (pi=[128, 128]) and critic (vf=[128, 128]) networks with Tanh activations mapping 8D observations to a 1D Gaussian setpoint distribution and scalar state-value estimate.",
    "configurationBadge": "100K Steps // CPU",
    "hyperparameters": [
      {
        "name": "total_timesteps",
        "value": "100,000",
        "description": "Total training environment interactions (~149 complete 7-day episodes)"
      },
      {
        "name": "learning_rate",
        "value": "0.0003",
        "description": "Adam optimizer step size for Actor-Critic parameter updates"
      },
      {
        "name": "n_steps / batch_size",
        "value": "2,048 / 64",
        "description": "Rollout buffer horizon (>3 days) and minibatch gradient update size"
      },
      {
        "name": "n_epochs",
        "value": "10",
        "description": "Gradient optimization passes executed per collected rollout batch"
      },
      {
        "name": "gamma / gae_lambda",
        "value": "0.99 / 0.95",
        "description": "Discount factor and Generalized Advantage Estimation variance-bias trade-off"
      },
      {
        "name": "clip_range / ent_coef",
        "value": "0.2 / 0.005",
        "description": "PPO surrogate ratio clipping epsilon and entropy exploration coefficient"
      },
      {
        "name": "reward_weights",
        "value": "w_e=1.0, w_c=3.5, w_s=0.2",
        "description": "Multi-objective penalties for energy, quadratic comfort violation, and stability"
      }
    ],
    "objectiveCriterion": "Clipped PPO Surrogate Policy Objective + Value MSE + Entropy Bonus",
    "optimizationSolver": "Adam Optimizer with Gradient Norm Clipping (max_grad_norm = 0.5)",
    "trainingProcess": "Training executes for 100,000 steps in 104.32 seconds on CPU. Running 20-episode reward rises from -6,281.46 at episode 1 to -234.29 at episode 149, while EvalCallback evaluates every 5,000 steps and archives the best checkpoint at -224.03 ± 2.98 reward."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Multi-Scenario Multi-Seed Deterministic Rollout",
    "description": "Evaluates the best deterministic PPO checkpoint across four scenarios (standard, heatwave, high_occupancy, weekend) averaged over three seeds (42, 101, 2024) against Rule-Based Scheduled, Occupancy-Aware, and Fixed 22.5 °C Thermostat baselines.",
    "metrics": [
      {
        "label": "COMFORT VIOLATION",
        "value": "7.92%",
        "benchmark": "Baseline: 100.0%"
      },
      {
        "label": "DEGREE HOURS",
        "value": "0.92 °C·h",
        "benchmark": "Baseline: 115.99"
      },
      {
        "label": "7-DAY ENERGY",
        "value": "270.17 kWh",
        "benchmark": "Fixed: 296.76"
      },
      {
        "label": "ENERGY SAVING",
        "value": "8.96%",
        "benchmark": "Target > 5.0%"
      },
      {
        "label": "WEEKEND VIOLATION",
        "value": "0.00%",
        "benchmark": "Baseline: 90.78%"
      },
      {
        "label": "EVAL REWARD",
        "value": "-221.81",
        "benchmark": "Baseline: -3581.5"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Overview & Cross-Scenario KPI Dashboard",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/smart_building_energy_control/image01.webp",
      "telemetryNote": "Summarizes 270.2 kWh 7-day PPO load, 7.9% comfort violation, and 4.39 kW peak demand across all benchmarked controllers."
    },
    {
      "title": "Interactive Scenario Trajectory Explorer",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/smart_building_energy_control/image02.webp",
      "telemetryNote": "Displays real-time 3-day indoor temperature curves proving PPO stays within the ASHRAE 55 band while Rule-Based drifts above 26 °C."
    },
    {
      "title": "PPO Training Convergence & Run Telemetry",
      "badge": "Learning Curve",
      "imageUrl": "/images/projects/smart_building_energy_control/image03.webp",
      "telemetryNote": "Tracks 149 training episodes (100,000 steps in 1.74 min) converging smoothly from -6,281 to -234.29 running reward."
    },
    {
      "title": "Energy & Thermal Comfort Analytics Grid",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/smart_building_energy_control/image04.webp",
      "telemetryNote": "Combines cumulative energy, comfort degree-hours, HVAC power dynamics, and Pareto scatter diagnostics in one view."
    },
    {
      "title": "1R1C Thermal & Reward Formulation Reference",
      "badge": "Methodology",
      "imageUrl": "/images/projects/smart_building_energy_control/image05.webp",
      "telemetryNote": "Renders the governing heat-balance ODE and multi-objective RL reward formulation with exact penalty weights."
    },
    {
      "title": "Training Curve, 7-Day Energy & Pareto Grid",
      "badge": "Multi-Panel",
      "imageUrl": "/images/projects/smart_building_energy_control/image06.webp",
      "telemetryNote": "Demonstrates PPO achieving lower energy than Fixed Thermostat across all four scenarios while occupying the bottom-left Pareto frontier."
    },
    {
      "title": "Indoor Temperature Profile: PPO vs Rule-Based",
      "badge": "Time Series",
      "imageUrl": "/images/projects/smart_building_energy_control/image07.webp",
      "telemetryNote": "Reveals PPO's anticipatory pre-cooling prior to 08:00 arrival, eliminating morning thermal lag spikes up to 27.3 °C."
    },
    {
      "title": "Diurnal Outdoor Temp vs Indoor Response & Power",
      "badge": "Thermal Dynamics",
      "imageUrl": "/images/projects/smart_building_energy_control/image08.webp",
      "telemetryNote": "Shows indoor temperature held within 23.0–25.8 °C against 35 °C outdoor peaks with controlled 4.39 kW afternoon chiller load."
    },
    {
      "title": "HVAC Setpoint Modulation & Power Behavior",
      "badge": "Control Policy",
      "imageUrl": "/images/projects/smart_building_energy_control/image09.webp",
      "telemetryNote": "Illustrates smooth continuous setpoint transitions from 26.0 °C setback down to 20.0 °C pre-cooling ahead of office occupancy."
    },
    {
      "title": "Seven-Day Commercial Office Occupancy Profile",
      "badge": "Input Profile",
      "imageUrl": "/images/projects/smart_building_energy_control/image10.webp",
      "telemetryNote": "Confirms realistic weekday attendance peaks (20–24 occupants), midday lunch dips, and 0–2 weekend staff schedules."
    },
    {
      "title": "Reward Component Decomposition Over Time",
      "badge": "Reward Analysis",
      "imageUrl": "/images/projects/smart_building_energy_control/image11.webp",
      "telemetryNote": "Verifies comfort penalties converge near zero after initial transient, leaving total reward driven by proportional cooling energy."
    },
    {
      "title": "Comfort Violation & Degree-Hours Comparison",
      "badge": "Benchmark Bar",
      "imageUrl": "/images/projects/smart_building_energy_control/image12.webp",
      "telemetryNote": "Highlights a 99.2% reduction in standard-week degree-hours (0.92 vs 115.99 °C·h) and 0.00% weekend violation."
    }
  ],
  "mlVisualizationTelemetryNote": "Training convergence on seed 42 (-234.29 running reward) aligns closely with held-out multi-seed evaluation (-221.81 standard return), confirming zero overfitting to initial conditions.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Policy inspection shows PPO autonomously learns anticipatory pre-cooling from cyclical time and occupancy cues, lowering HVAC setpoints ahead of 08:00 arrivals to counteract 1R1C thermal mass lag while relaxing cooling overnight and throughout weekends.",
    "keyDrivers": [
      {
        "factor": "Pre-Cooling Before Work Hours",
        "impact": "Reduces standard-week comfort violation from 100.0% (Rule-Based) to 7.92% and degree-hours by 99.2%."
      },
      {
        "factor": "Off-Hour & Weekend Setback",
        "impact": "Cuts 7-day electricity use by 8.96% (26.59 kWh) compared with 24/7 Fixed Thermostat operation."
      },
      {
        "factor": "Quadratic Comfort Weighting (w_c = 3.5)",
        "impact": "Drives comfort penalties near zero during occupied shifts while optimizing residual energy cost."
      }
    ],
    "limitations": [
      {
        "title": "Single-Zone 1R1C Lumped Approximation",
        "description": "Assumes uniform room air temperature and omits multi-zone thermal coupling, spatial gradients, and EnergyPlus building geometry."
      },
      {
        "title": "Synthetic Weather & Unmodeled IAQ",
        "description": "Uses analytical diurnal/synoptic weather waves rather than EPW weather files and excludes relative humidity and CO₂ ventilation control."
      },
      {
        "title": "Heatwave Capacity Saturation",
        "description": "Under extreme 38.5 °C heatwaves, the 12 kW chiller limit causes 66.71% comfort violation even with optimal PPO scheduling."
      }
    ],
    "futureImprovements": [
      {
        "title": "EnergyPlus / Sinergym Co-Simulation",
        "target": "Validate PPO transferability against high-fidelity multi-zone IDF models and real EPW weather data."
      },
      {
        "title": "Indoor Air Quality & Humidity Control",
        "target": "Expand action space to modulate fresh-air damper rates keeping CO₂ below 800 ppm alongside thermal comfort."
      },
      {
        "title": "Time-of-Use Tariff & Storage Optimization",
        "target": "Integrate dynamic electricity pricing and thermal energy storage for off-peak load shifting."
      }
    ],
    "discoveryNote": "Direct physical deployment requires a hard safety override layer constraining compressor cycling rates and calibrating 1R1C parameters against sensor logs."
  },
  "mlModelArtifacts": [
    {
      "name": "ppo_smart_building.zip",
      "format": "SB3 ZIP"
    },
    {
      "name": "best_model.zip",
      "format": "SB3 ZIP"
    },
    {
      "name": "ppo_final.zip",
      "format": "SB3 ZIP"
    },
    {
      "name": "evaluations.npz",
      "format": "NPZ"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Reinforcement Learning",
    "modelArchitecture": "PPO Actor-Critic MLP",
    "coreFramework": "Stable-Baselines3 PyTorch",
    "validationMethod": "Multi-Seed Scenario Rollout"
  }
}
```
