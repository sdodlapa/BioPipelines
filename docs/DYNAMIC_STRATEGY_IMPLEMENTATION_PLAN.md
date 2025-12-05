# Dynamic Strategy Selection Implementation Plan
## BioPipelines Multi-Model Orchestration Enhancement

**Created:** December 5, 2025  
**Status:** Planning  
**Priority:** High  
**Estimated Effort:** 5-10 days (phased)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Gap Analysis](#gap-analysis)
4. [Proposed Architecture](#proposed-architecture)
5. [Implementation Phases](#implementation-phases)
6. [Detailed Component Specifications](#detailed-component-specifications)
7. [Integration Points](#integration-points)
8. [Testing Strategy](#testing-strategy)
9. [Risk Assessment](#risk-assessment)
10. [Research Topics for Further Exploration](#research-topics-for-further-exploration)

---

## Executive Summary

### Goal

Implement a **dynamic strategy selection system** that:
1. Detects available hardware resources (GPUs, SLURM partitions, cloud APIs)
2. Loads pre-configured strategy profiles from YAML files
3. Allows users to select or override strategies at session start
4. Routes requests to appropriate models based on the active strategy

### Key Principle

> **"Configure once, run anywhere"** - The same BioPipelines codebase should work optimally whether deployed on 10× T4s, a single H100, or cloud-only mode.

### Success Criteria

| Metric | Target |
|--------|--------|
| Strategy selection time | < 2 seconds at session start |
| Profile switching | Zero code changes required |
| Fallback reliability | 99.9% request success rate |
| New profile creation | < 30 minutes by non-developer |

---

## Current State Analysis

### What Exists ✅

```
src/workflow_composer/
├── llm/
│   ├── orchestrator.py          # ModelOrchestrator with Strategy enum
│   ├── strategies.py            # Strategy, StrategyConfig, PRESETS
│   ├── task_router.py           # TaskRouter with task classification
│   └── providers.py             # LocalProvider, CloudProvider
│
├── providers/
│   ├── router.py                # ProviderRouter (cloud cascade)
│   ├── t4_router.py             # T4ModelRouter (task-based routing)
│   ├── local_model_registry.py  # Model catalog access
│   └── registry.py              # ProviderRegistry
│
└── config/
    └── local_model_catalog.yaml # Model definitions with T4 compatibility
```

### Current Flow

```
User Request
     │
     ▼
ModelOrchestrator.__init__(strategy=Strategy.LOCAL_FIRST)  ← Fixed at startup
     │
     ├── Strategy determines: local vs cloud preference
     │
     ▼
LocalProvider or CloudProvider
     │
     ▼
Response
```

**Problem**: Strategy is fixed at `__init__`, no hardware detection, no profile loading.

---

## Gap Analysis

### Missing Components

| Component | Purpose | Priority | Effort |
|-----------|---------|----------|--------|
| **ResourceDetector** | Detect GPUs, SLURM, cloud APIs | P0 | 1 day |
| **StrategyProfile** | YAML-defined routing configuration | P0 | 1 day |
| **StrategySelector** | Match resources → best profile | P0 | 1 day |
| **UnifiedRouter** | Single entry point for all routing | P1 | 2 days |
| **SessionManager** | Manage active strategy state | P1 | 1 day |
| **Strategy CLI** | Interactive strategy selection | P2 | 0.5 days |

### Integration Gaps

1. `T4ModelRouter` and `ProviderRouter` are not connected
2. No SLURM partition detection
3. No cloud API availability checking at startup
4. Strategy profiles are Python dicts (PRESETS), not external YAML

---

## Proposed Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SESSION INITIALIZATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ ResourceDetector │───▶│ StrategySelector │───▶│  SessionManager  │       │
│  │                  │    │                  │    │                  │       │
│  │ - GPU detection  │    │ - Load profiles  │    │ - Active profile │       │
│  │ - SLURM check    │    │ - Match resources│    │ - Runtime state  │       │
│  │ - API key check  │    │ - User override  │    │ - Metrics        │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (StrategyProfile)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REQUEST ROUTING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        UnifiedRouter                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Task        │  │ Route       │  │ Execute     │  │ Fallback    │  │   │
│  │  │ Classifier  │─▶│ Selector    │─▶│ Request     │─▶│ Handler     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   T4 vLLM Fleet  │  │   Cloud APIs     │  │   H100/L4 Local  │          │
│  │   (T4ModelRouter)│  │   (ProviderRouter)│  │   (LocalProvider)│          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure (New Files)

```
src/workflow_composer/
├── strategy/                    # NEW: Strategy selection module
│   ├── __init__.py
│   ├── resource_detector.py     # Hardware/API detection
│   ├── profile.py               # StrategyProfile dataclass
│   ├── selector.py              # StrategySelector
│   ├── session.py               # SessionManager
│   └── unified_router.py        # UnifiedRouter
│
config/
├── strategies/                  # NEW: Strategy profile YAML files
│   ├── t4_hybrid.yaml           # 10× T4 + DeepSeek fallback
│   ├── h100_local.yaml          # Single H100, all local
│   ├── l4_t4_combined.yaml      # 4× L4 + 4× T4
│   ├── cloud_only.yaml          # No local GPUs
│   └── development.yaml         # Minimal, fast iteration
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)

**Goal**: Build foundational components without breaking existing code.

#### 1.1 ResourceDetector

```python
# src/workflow_composer/strategy/resource_detector.py

@dataclass
class ResourceProfile:
    """Detected hardware and API resources."""
    
    # GPU Information
    gpu_available: bool
    gpu_type: Optional[str]  # "T4", "L4", "H100", "A100", None
    gpu_count: int
    gpu_memory_gb: float
    
    # SLURM Information
    slurm_available: bool
    slurm_partitions: List[str]  # ["t4flex", "h100flex", "a100flex"]
    current_partition: Optional[str]
    
    # Cloud API Availability
    cloud_apis: Dict[str, bool]  # {"deepseek": True, "openai": False, ...}
    
    # System Information
    hostname: str
    cpu_count: int
    memory_gb: float
    
    # Derived
    @property
    def deployment_mode(self) -> str:
        """Infer deployment mode: 'slurm', 'local', 'cloud_only'."""
        if self.slurm_available:
            return "slurm"
        elif self.gpu_available:
            return "local"
        else:
            return "cloud_only"


class ResourceDetector:
    """Detects available hardware and cloud resources."""
    
    def detect(self) -> ResourceProfile:
        """Run all detection methods and return profile."""
        return ResourceProfile(
            gpu_available=self._detect_gpu(),
            gpu_type=self._detect_gpu_type(),
            gpu_count=self._detect_gpu_count(),
            gpu_memory_gb=self._detect_gpu_memory(),
            slurm_available=self._detect_slurm(),
            slurm_partitions=self._detect_slurm_partitions(),
            current_partition=self._detect_current_partition(),
            cloud_apis=self._detect_cloud_apis(),
            hostname=socket.gethostname(),
            cpu_count=os.cpu_count() or 1,
            memory_gb=self._detect_system_memory(),
        )
    
    def _detect_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False
    
    def _detect_gpu_type(self) -> Optional[str]:
        """Detect GPU type (T4, L4, H100, etc.)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip().split("\n")[0]
                # Parse common GPU types
                if "T4" in gpu_name:
                    return "T4"
                elif "L4" in gpu_name:
                    return "L4"
                elif "H100" in gpu_name:
                    return "H100"
                elif "A100" in gpu_name:
                    return "A100"
                elif "V100" in gpu_name:
                    return "V100"
                return gpu_name  # Return full name if not recognized
        except Exception:
            pass
        return None
    
    def _detect_slurm(self) -> bool:
        """Check if running in SLURM environment."""
        return "SLURM_JOB_ID" in os.environ or shutil.which("squeue") is not None
    
    def _detect_slurm_partitions(self) -> List[str]:
        """Get available SLURM partitions."""
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return [p.strip().rstrip("*") for p in result.stdout.strip().split("\n") if p.strip()]
        except Exception:
            pass
        return []
    
    def _detect_cloud_apis(self) -> Dict[str, bool]:
        """Check which cloud APIs have keys configured."""
        return {
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "cerebras": bool(os.getenv("CEREBRAS_API_KEY")),
        }
```

#### 1.2 StrategyProfile

```python
# src/workflow_composer/strategy/profile.py

@dataclass
class ModelRoute:
    """Routing configuration for a single task category."""
    model: str                    # Model ID or HuggingFace path
    local: bool                   # True = use local vLLM, False = use cloud
    quantization: Optional[str]   # "fp16", "int8", "int4"
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Fallback chain
    fallback_model: Optional[str] = None
    fallback_local: bool = False


@dataclass  
class StrategyProfile:
    """Complete strategy configuration loaded from YAML."""
    
    # Metadata
    name: str
    description: str
    version: str = "1.0"
    
    # Hardware requirements
    min_gpu_memory_gb: float = 0
    required_gpu_types: List[str] = field(default_factory=list)  # Empty = any
    requires_slurm: bool = False
    required_cloud_apis: List[str] = field(default_factory=list)
    
    # Task routing configuration
    routes: Dict[str, ModelRoute] = field(default_factory=dict)
    
    # Default fallback (used when task-specific fallback not defined)
    default_fallback_model: str = "deepseek-v3"
    default_fallback_local: bool = False
    
    # Behavior settings
    prefer_local: bool = True
    max_retries: int = 2
    timeout_seconds: float = 60.0
    enable_caching: bool = True
    
    # Cost controls
    max_cost_per_request: Optional[float] = None
    monthly_budget_limit: Optional[float] = None
    
    def matches_resources(self, resources: ResourceProfile) -> Tuple[bool, str]:
        """Check if this profile is compatible with detected resources."""
        # Check GPU memory
        if resources.gpu_memory_gb < self.min_gpu_memory_gb:
            return False, f"Insufficient GPU memory: {resources.gpu_memory_gb}GB < {self.min_gpu_memory_gb}GB"
        
        # Check GPU type
        if self.required_gpu_types and resources.gpu_type not in self.required_gpu_types:
            return False, f"GPU type {resources.gpu_type} not in {self.required_gpu_types}"
        
        # Check SLURM
        if self.requires_slurm and not resources.slurm_available:
            return False, "SLURM required but not available"
        
        # Check cloud APIs
        for api in self.required_cloud_apis:
            if not resources.cloud_apis.get(api, False):
                return False, f"Cloud API '{api}' not configured"
        
        return True, "Compatible"
    
    def get_route(self, task: str) -> ModelRoute:
        """Get routing configuration for a task category."""
        if task in self.routes:
            return self.routes[task]
        # Return default route
        return ModelRoute(
            model=self.default_fallback_model,
            local=self.default_fallback_local,
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "StrategyProfile":
        """Load profile from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Parse routes
        routes = {}
        for task, route_data in data.get("routes", {}).items():
            routes[task] = ModelRoute(**route_data)
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            min_gpu_memory_gb=data.get("hardware", {}).get("min_gpu_memory_gb", 0),
            required_gpu_types=data.get("hardware", {}).get("gpu_types", []),
            requires_slurm=data.get("hardware", {}).get("requires_slurm", False),
            required_cloud_apis=data.get("hardware", {}).get("required_cloud_apis", []),
            routes=routes,
            default_fallback_model=data.get("fallback", {}).get("model", "deepseek-v3"),
            default_fallback_local=data.get("fallback", {}).get("local", False),
            prefer_local=data.get("behavior", {}).get("prefer_local", True),
            max_retries=data.get("behavior", {}).get("max_retries", 2),
            timeout_seconds=data.get("behavior", {}).get("timeout_seconds", 60.0),
            enable_caching=data.get("behavior", {}).get("enable_caching", True),
            max_cost_per_request=data.get("cost", {}).get("max_per_request"),
            monthly_budget_limit=data.get("cost", {}).get("monthly_limit"),
        )
```

#### 1.3 Strategy Profile YAML Files

```yaml
# config/strategies/t4_hybrid.yaml
name: "T4 Hybrid"
description: "10× T4 GPUs locally + DeepSeek cloud fallback"
version: "1.0"

hardware:
  min_gpu_memory_gb: 16
  gpu_types: ["T4"]
  requires_slurm: true
  required_cloud_apis: ["deepseek"]

routes:
  intent_parsing:
    model: "meta-llama/Llama-3.2-3B-Instruct"
    local: true
    quantization: "fp16"
    max_tokens: 2048
    fallback_model: "deepseek-v3"
    fallback_local: false
  
  code_generation:
    model: "Qwen/Qwen2.5-Coder-7B-Instruct"
    local: true
    quantization: "int8"
    max_tokens: 4096
    fallback_model: "deepseek-v3"
    fallback_local: false
  
  code_validation:
    model: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    local: true
    quantization: "fp16"
    max_tokens: 2048
  
  data_analysis:
    model: "microsoft/Phi-3.5-mini-instruct"
    local: true
    quantization: "fp16"
    max_tokens: 4096
  
  math_statistics:
    model: "Qwen/Qwen2.5-Math-7B-Instruct"
    local: true
    quantization: "int8"
    max_tokens: 2048
  
  biomedical:
    model: "BioMistral/BioMistral-7B"
    local: true
    quantization: "int8"
    fallback_model: "claude-3.5-sonnet"
    fallback_local: false
  
  documentation:
    model: "google/gemma-2-9b-it"
    local: true
    quantization: "int8"
    max_tokens: 8192
  
  embeddings:
    model: "BAAI/bge-m3"
    local: true
    quantization: "fp16"
  
  safety:
    model: "meta-llama/Llama-Guard-3-1B"
    local: true
    quantization: "fp16"
  
  orchestration:
    model: "deepseek-v3"
    local: false  # Cloud only - too complex for small models

fallback:
  model: "deepseek-v3"
  local: false

behavior:
  prefer_local: true
  max_retries: 2
  timeout_seconds: 60
  enable_caching: true

cost:
  max_per_request: 0.10
  monthly_limit: 50.00
```

```yaml
# config/strategies/cloud_only.yaml
name: "Cloud Only"
description: "No local GPUs - all requests go to cloud APIs"
version: "1.0"

hardware:
  min_gpu_memory_gb: 0
  gpu_types: []
  requires_slurm: false
  required_cloud_apis: ["deepseek"]

routes:
  intent_parsing:
    model: "deepseek-v3"
    local: false
  
  code_generation:
    model: "deepseek-v3"
    local: false
  
  code_validation:
    model: "deepseek-v3"
    local: false
  
  data_analysis:
    model: "deepseek-v3"
    local: false
  
  math_statistics:
    model: "deepseek-v3"
    local: false
  
  biomedical:
    model: "claude-3.5-sonnet"
    local: false
  
  documentation:
    model: "claude-3.5-sonnet"
    local: false
  
  embeddings:
    model: "openai/text-embedding-3-small"
    local: false
  
  safety:
    model: "deepseek-v3"
    local: false
  
  orchestration:
    model: "deepseek-v3"
    local: false

fallback:
  model: "gpt-4o"
  local: false

behavior:
  prefer_local: false
  max_retries: 3
  timeout_seconds: 120
  enable_caching: true

cost:
  max_per_request: 1.00
  monthly_limit: 200.00
```

### Phase 2: Strategy Selection (Days 3-4)

#### 2.1 StrategySelector

```python
# src/workflow_composer/strategy/selector.py

class StrategySelector:
    """Selects optimal strategy profile based on resources and user preferences."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or Path(__file__).parent.parent.parent.parent / "config" / "strategies"
        self.detector = ResourceDetector()
        self.profiles: Dict[str, StrategyProfile] = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load all strategy profiles from YAML files."""
        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")
            return
        
        for yaml_file in self.profiles_dir.glob("*.yaml"):
            try:
                profile = StrategyProfile.from_yaml(yaml_file)
                self.profiles[profile.name.lower().replace(" ", "_")] = profile
                logger.info(f"Loaded strategy profile: {profile.name}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
    
    def detect_resources(self) -> ResourceProfile:
        """Detect current hardware resources."""
        return self.detector.detect()
    
    def find_compatible_profiles(self, resources: ResourceProfile) -> List[Tuple[str, StrategyProfile, str]]:
        """Find all profiles compatible with current resources."""
        compatible = []
        for name, profile in self.profiles.items():
            matches, reason = profile.matches_resources(resources)
            if matches:
                compatible.append((name, profile, reason))
        return compatible
    
    def recommend_profile(self, resources: ResourceProfile) -> Tuple[str, StrategyProfile]:
        """Recommend the best profile for current resources."""
        compatible = self.find_compatible_profiles(resources)
        
        if not compatible:
            # No compatible profiles - use cloud_only as ultimate fallback
            if "cloud_only" in self.profiles:
                return "cloud_only", self.profiles["cloud_only"]
            raise RuntimeError("No compatible strategy profiles found")
        
        # Priority order for recommendation
        priority = ["h100_local", "l4_t4_combined", "t4_hybrid", "cloud_only", "development"]
        
        for pname in priority:
            for name, profile, _ in compatible:
                if name == pname:
                    return name, profile
        
        # Return first compatible
        return compatible[0][0], compatible[0][1]
    
    def select(
        self,
        override_profile: Optional[str] = None,
        interactive: bool = False,
    ) -> Tuple[str, StrategyProfile, ResourceProfile]:
        """
        Select a strategy profile.
        
        Args:
            override_profile: Force use of specific profile
            interactive: Prompt user for confirmation/selection
        
        Returns:
            Tuple of (profile_name, StrategyProfile, ResourceProfile)
        """
        resources = self.detect_resources()
        
        # If override specified, use it
        if override_profile:
            if override_profile not in self.profiles:
                available = ", ".join(self.profiles.keys())
                raise ValueError(f"Unknown profile: {override_profile}. Available: {available}")
            profile = self.profiles[override_profile]
            matches, reason = profile.matches_resources(resources)
            if not matches:
                logger.warning(f"Profile '{override_profile}' may not be compatible: {reason}")
            return override_profile, profile, resources
        
        # Recommend best profile
        recommended_name, recommended_profile = self.recommend_profile(resources)
        
        if interactive:
            return self._interactive_select(resources, recommended_name, recommended_profile)
        
        return recommended_name, recommended_profile, resources
    
    def _interactive_select(
        self,
        resources: ResourceProfile,
        recommended_name: str,
        recommended_profile: StrategyProfile,
    ) -> Tuple[str, StrategyProfile, ResourceProfile]:
        """Interactive profile selection."""
        print("\n" + "="*60)
        print("BioPipelines Strategy Selection")
        print("="*60)
        
        print(f"\nDetected Resources:")
        print(f"  GPU: {resources.gpu_type or 'None'} × {resources.gpu_count}")
        print(f"  GPU Memory: {resources.gpu_memory_gb:.1f} GB")
        print(f"  SLURM: {'Yes' if resources.slurm_available else 'No'}")
        print(f"  Cloud APIs: {[k for k, v in resources.cloud_apis.items() if v]}")
        
        print(f"\nRecommended: {recommended_name} ({recommended_profile.description})")
        
        compatible = self.find_compatible_profiles(resources)
        print(f"\nCompatible profiles ({len(compatible)}):")
        for i, (name, profile, _) in enumerate(compatible, 1):
            marker = "→" if name == recommended_name else " "
            print(f"  {marker} [{i}] {name}: {profile.description}")
        
        print(f"\nPress Enter to use '{recommended_name}', or type profile name/number:")
        
        try:
            choice = input().strip()
        except EOFError:
            choice = ""
        
        if not choice:
            return recommended_name, recommended_profile, resources
        
        # Try as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(compatible):
                name = compatible[idx][0]
                return name, compatible[idx][1], resources
        except ValueError:
            pass
        
        # Try as name
        if choice in self.profiles:
            return choice, self.profiles[choice], resources
        
        print(f"Unknown selection '{choice}', using recommended.")
        return recommended_name, recommended_profile, resources
```

#### 2.2 SessionManager

```python
# src/workflow_composer/strategy/session.py

@dataclass
class SessionState:
    """Runtime state for the current session."""
    profile_name: str
    profile: StrategyProfile
    resources: ResourceProfile
    started_at: datetime
    request_count: int = 0
    total_cost: float = 0.0
    errors: int = 0
    
    # Model availability cache (updated by health checks)
    model_health: Dict[str, bool] = field(default_factory=dict)


class SessionManager:
    """Manages the active strategy session."""
    
    _instance: Optional["SessionManager"] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._state: Optional[SessionState] = None
        self._selector = StrategySelector()
    
    @property
    def is_active(self) -> bool:
        """Check if a session is active."""
        return self._state is not None
    
    @property
    def state(self) -> SessionState:
        """Get current session state."""
        if not self._state:
            raise RuntimeError("No active session. Call start_session() first.")
        return self._state
    
    @property
    def profile(self) -> StrategyProfile:
        """Get current strategy profile."""
        return self.state.profile
    
    def start_session(
        self,
        profile_override: Optional[str] = None,
        interactive: bool = False,
    ) -> SessionState:
        """
        Start a new strategy session.
        
        Args:
            profile_override: Force specific profile
            interactive: Prompt for selection
        """
        profile_name, profile, resources = self._selector.select(
            override_profile=profile_override,
            interactive=interactive,
        )
        
        self._state = SessionState(
            profile_name=profile_name,
            profile=profile,
            resources=resources,
            started_at=datetime.now(),
        )
        
        logger.info(f"Session started with profile: {profile_name}")
        return self._state
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        if not self._state:
            return {}
        
        summary = {
            "profile": self._state.profile_name,
            "duration_seconds": (datetime.now() - self._state.started_at).total_seconds(),
            "requests": self._state.request_count,
            "total_cost": self._state.total_cost,
            "errors": self._state.errors,
        }
        
        self._state = None
        logger.info(f"Session ended: {summary}")
        return summary
    
    def record_request(self, cost: float = 0.0, error: bool = False):
        """Record a request in the session."""
        if self._state:
            self._state.request_count += 1
            self._state.total_cost += cost
            if error:
                self._state.errors += 1


# Global accessor
def get_session() -> SessionManager:
    """Get the global session manager."""
    return SessionManager()
```

### Phase 3: Unified Routing (Days 5-6)

#### 3.1 UnifiedRouter

```python
# src/workflow_composer/strategy/unified_router.py

class UnifiedRouter:
    """
    Single entry point for all model routing.
    
    Combines T4ModelRouter, ProviderRouter, and LocalProvider
    under a unified interface driven by the active StrategyProfile.
    """
    
    def __init__(self, session: Optional[SessionManager] = None):
        self.session = session or get_session()
        
        # Lazy-loaded routers
        self._t4_router: Optional[T4ModelRouter] = None
        self._provider_router: Optional[ProviderRouter] = None
        self._local_provider: Optional[LocalProvider] = None
    
    @property
    def profile(self) -> StrategyProfile:
        """Get active strategy profile."""
        return self.session.profile
    
    @property
    def t4_router(self) -> T4ModelRouter:
        """Get or create T4 router."""
        if self._t4_router is None:
            self._t4_router = T4ModelRouter()
        return self._t4_router
    
    @property
    def provider_router(self) -> ProviderRouter:
        """Get or create provider router."""
        if self._provider_router is None:
            self._provider_router = ProviderRouter()
        return self._provider_router
    
    async def route(
        self,
        task: str,
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route a request based on task category and active profile.
        
        Args:
            task: Task category (intent_parsing, code_generation, etc.)
            prompt: The prompt to process
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Response dictionary with content, model, latency, cost, etc.
        """
        route_config = self.profile.get_route(task)
        
        # Merge kwargs with route config defaults
        params = {
            "max_tokens": route_config.max_tokens,
            "temperature": route_config.temperature,
            **kwargs,
        }
        
        try:
            if route_config.local:
                result = await self._route_local(task, prompt, route_config, params)
            else:
                result = await self._route_cloud(prompt, route_config, params)
            
            self.session.record_request(cost=result.get("cost", 0))
            return result
            
        except Exception as e:
            logger.error(f"Primary route failed for {task}: {e}")
            
            # Try fallback
            if route_config.fallback_model:
                try:
                    if route_config.fallback_local:
                        result = await self._route_local_model(
                            route_config.fallback_model, prompt, params
                        )
                    else:
                        result = await self._route_cloud_model(
                            route_config.fallback_model, prompt, params
                        )
                    result["fallback_used"] = True
                    self.session.record_request(cost=result.get("cost", 0))
                    return result
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
            
            self.session.record_request(error=True)
            raise
    
    async def _route_local(
        self,
        task: str,
        prompt: str,
        route_config: ModelRoute,
        params: Dict,
    ) -> Dict[str, Any]:
        """Route to local vLLM server."""
        # Map task to T4ModelRouter category
        task_category = self._map_task_to_category(task)
        
        result = await self.t4_router.complete(
            task=task_category,
            prompt=prompt,
            **params,
        )
        
        result["route_type"] = "local"
        result["model_config"] = route_config
        return result
    
    async def _route_cloud(
        self,
        prompt: str,
        route_config: ModelRoute,
        params: Dict,
    ) -> Dict[str, Any]:
        """Route to cloud provider."""
        response = self.provider_router.complete(
            prompt=prompt,
            preferred_model=route_config.model,
            **params,
        )
        
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "cost": response.cost or 0,
            "latency_ms": response.latency_ms,
            "route_type": "cloud",
        }
    
    def _map_task_to_category(self, task: str) -> str:
        """Map profile task names to T4ModelRouter categories."""
        mapping = {
            "intent_parsing": "intent",
            "code_generation": "codegen",
            "code_validation": "validation",
            "data_analysis": "analysis",
            "math_statistics": "math",
            "biomedical": "biomedical",
            "documentation": "docs",
            "embeddings": "embeddings",
            "safety": "safety",
            "orchestration": "orchestration",
        }
        return mapping.get(task, task)
    
    # Convenience methods
    async def complete(self, prompt: str, task: str = "general", **kwargs):
        """Simple completion with auto task classification."""
        return await self.route(task, prompt, **kwargs)
    
    async def embed(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Generate embeddings."""
        return await self.route("embeddings", str(texts) if isinstance(texts, str) else "\n".join(texts))
```

### Phase 4: Integration & CLI (Days 7-8)

#### 4.1 CLI Commands

```python
# src/workflow_composer/cli/strategy_cli.py

import click
from ..strategy import get_session, StrategySelector, ResourceDetector

@click.group()
def strategy():
    """Strategy management commands."""
    pass

@strategy.command()
def detect():
    """Detect available hardware resources."""
    detector = ResourceDetector()
    resources = detector.detect()
    
    click.echo("\n" + "="*50)
    click.echo("Detected Resources")
    click.echo("="*50)
    click.echo(f"GPU Available: {resources.gpu_available}")
    click.echo(f"GPU Type: {resources.gpu_type or 'N/A'}")
    click.echo(f"GPU Count: {resources.gpu_count}")
    click.echo(f"GPU Memory: {resources.gpu_memory_gb:.1f} GB")
    click.echo(f"SLURM Available: {resources.slurm_available}")
    click.echo(f"SLURM Partitions: {', '.join(resources.slurm_partitions) or 'N/A'}")
    click.echo(f"Deployment Mode: {resources.deployment_mode}")
    click.echo("\nCloud APIs Configured:")
    for api, available in resources.cloud_apis.items():
        status = "✓" if available else "✗"
        click.echo(f"  {status} {api}")

@strategy.command()
def list():
    """List available strategy profiles."""
    selector = StrategySelector()
    
    click.echo("\n" + "="*50)
    click.echo("Available Strategy Profiles")
    click.echo("="*50)
    
    for name, profile in selector.profiles.items():
        click.echo(f"\n{name}:")
        click.echo(f"  Description: {profile.description}")
        click.echo(f"  Min GPU Memory: {profile.min_gpu_memory_gb} GB")
        click.echo(f"  GPU Types: {profile.required_gpu_types or 'Any'}")
        click.echo(f"  Requires SLURM: {profile.requires_slurm}")

@strategy.command()
@click.option("--profile", "-p", help="Force specific profile")
@click.option("--interactive", "-i", is_flag=True, help="Interactive selection")
def start(profile, interactive):
    """Start a strategy session."""
    session = get_session()
    state = session.start_session(
        profile_override=profile,
        interactive=interactive,
    )
    
    click.echo(f"\n✓ Session started with profile: {state.profile_name}")
    click.echo(f"  Description: {state.profile.description}")

@strategy.command()
def status():
    """Show current session status."""
    session = get_session()
    if not session.is_active:
        click.echo("No active session.")
        return
    
    state = session.state
    click.echo("\n" + "="*50)
    click.echo("Current Session")
    click.echo("="*50)
    click.echo(f"Profile: {state.profile_name}")
    click.echo(f"Started: {state.started_at}")
    click.echo(f"Requests: {state.request_count}")
    click.echo(f"Total Cost: ${state.total_cost:.4f}")
    click.echo(f"Errors: {state.errors}")
```

---

## Integration Points

### 1. Modify ModelOrchestrator

```python
# In src/workflow_composer/llm/orchestrator.py

class ModelOrchestrator:
    def __init__(
        self,
        strategy: Strategy = Strategy.AUTO,
        config: Optional[StrategyConfig] = None,
        use_unified_router: bool = True,  # NEW
        ...
    ):
        if use_unified_router:
            # Use new strategy-aware routing
            self._router = UnifiedRouter()
        else:
            # Legacy mode
            self.local = local_provider or LocalProvider()
            self.cloud = cloud_provider or CloudProvider()
```

### 2. Update Chat Agent Initialization

```python
# In chat agent startup code

from workflow_composer.strategy import get_session

def initialize_agent():
    # Start strategy session (with interactive selection)
    session = get_session()
    session.start_session(interactive=True)
    
    # Now create orchestrator - it will use the active session
    orchestrator = ModelOrchestrator()
```

---

## Testing Strategy

### Unit Tests

```python
# tests/strategy/test_resource_detector.py

def test_gpu_detection_with_gpu(mocker):
    mocker.patch("subprocess.run", return_value=Mock(
        returncode=0,
        stdout="Tesla T4\n"
    ))
    detector = ResourceDetector()
    resources = detector.detect()
    assert resources.gpu_available
    assert resources.gpu_type == "T4"

def test_gpu_detection_without_gpu(mocker):
    mocker.patch("subprocess.run", side_effect=FileNotFoundError)
    detector = ResourceDetector()
    resources = detector.detect()
    assert not resources.gpu_available
```

### Integration Tests

```python
# tests/strategy/test_integration.py

@pytest.mark.integration
def test_full_session_flow():
    session = get_session()
    
    # Start session
    state = session.start_session(profile_override="t4_hybrid")
    assert state.profile_name == "t4_hybrid"
    
    # Create router
    router = UnifiedRouter(session)
    
    # Route a request (mocked)
    result = await router.route("intent_parsing", "Test prompt")
    assert "content" in result
    
    # End session
    summary = session.end_session()
    assert summary["requests"] == 1
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Profile YAML syntax errors | Medium | High | Validate on load, provide schema |
| GPU detection fails | Low | Medium | Graceful fallback to cloud_only |
| Session state corruption | Low | High | Singleton pattern, clear state methods |
| Performance overhead | Low | Low | Lazy loading, caching |
| Backwards compatibility | Medium | Medium | Keep legacy mode as fallback |

---

## Research Topics for Further Exploration

### 1. **LiteLLM Integration**

LiteLLM provides a battle-tested routing layer. Consider:
- Using LiteLLM as the underlying router instead of ProviderRouter
- Benefits: 100+ provider support, built-in retries, cost tracking
- Concern: Additional dependency, may be overkill

**Research questions:**
- Can LiteLLM route to local vLLM servers?
- How does LiteLLM handle task-based routing (not just load balancing)?

### 2. **Model Context Protocol (MCP)**

Anthropic's MCP allows LLMs to access external tools/resources. For BioPipelines:
- Could MCP servers expose bioinformatics tools directly?
- Strategy profiles could define which MCP servers to connect

**Research:**
- https://modelcontextprotocol.io/
- Is MCP mature enough for production?

### 3. **Semantic Router**

[Semantic Router](https://github.com/aurelio-labs/semantic-router) by Aurelio Labs:
- Uses embeddings to classify queries into categories
- Could replace keyword-based task classification
- More robust to variations in user input

**Integration idea:**
```python
from semantic_router import Route, SemanticRouter

intent_route = Route(name="intent", utterances=["parse this query", "understand what I mean", ...])
code_route = Route(name="code", utterances=["write a script", "generate code", ...])

router = SemanticRouter(routes=[intent_route, code_route])
task = router(query)  # Returns "intent" or "code"
```

### 4. **Mixture of Experts (MoE) Patterns**

Modern MoE models (DeepSeek-V3, Mixtral) use sparse expert routing. Apply same concept:
- Each "expert" is a specialized small model
- Router selects 1-2 experts per query
- Reduces compute while maintaining quality

**Research:**
- How do MoE models decide which experts to activate?
- Can we learn routing weights from query patterns?

### 5. **Cost-Aware Routing (RouteLLM)**

[RouteLLM](https://github.com/lm-sys/RouteLLM) by LMSYS:
- Routes queries between strong (expensive) and weak (cheap) models
- Learns routing from preference data
- Could reduce costs 50%+ with minimal quality loss

**Integration idea:**
```python
from routellm import RouteLLMController

controller = RouteLLMController(
    strong_model="deepseek-v3",
    weak_model="llama-3.2-3b",
    threshold=0.5  # Route to strong if confidence < 0.5
)

response = controller.chat(messages)  # Auto-selects model
```

### 6. **Speculative Decoding**

Use small model to generate draft, large model to verify:
- Small local model (Llama-3.2-3B) generates candidate tokens
- Large cloud model (DeepSeek-V3) accepts/rejects in batch
- Reduces latency while maintaining quality

**BioPipelines application:**
- Use T4 models for initial generation
- Route to cloud for verification only when needed

### 7. **Adaptive Batch Sizing**

Dynamic batching based on:
- Current GPU memory usage
- Queue depth
- Time-of-day patterns

**Research:**
- vLLM already does continuous batching - can we tune it?
- SLURM job priority vs batch size tradeoffs

### 8. **Federated Model Selection**

If you scale to multiple universities/clusters:
- Each site has different hardware
- Central registry of model capabilities
- Query routes to nearest capable site

**Probably overkill for now**, but interesting for future.

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Core (Phase 1-2) | ResourceDetector, StrategyProfile, Selector |
| 2 | Routing (Phase 3) | UnifiedRouter, Session integration |
| 3 | Polish (Phase 4) | CLI, tests, documentation |
| 4 | Research | Evaluate LiteLLM, RouteLLM, SemanticRouter |

---

## Next Steps

1. **Review this plan** - Discuss with team, identify concerns
2. **Create Phase 1 skeleton** - Empty files, interfaces
3. **Implement ResourceDetector** - Most foundational piece
4. **Test on actual cluster** - Verify SLURM/GPU detection works
5. **Research deep-dive** - Pick 1-2 research topics to prototype

---

## Appendix: Quick Reference

### Profile Selection Decision Tree

```
START
  │
  ├── User specified --profile?
  │     └── YES → Use that profile
  │
  ├── H100 detected?
  │     └── YES → h100_local
  │
  ├── L4 + T4 detected?
  │     └── YES → l4_t4_combined
  │
  ├── T4 detected + DeepSeek API?
  │     └── YES → t4_hybrid
  │
  ├── Any cloud API configured?
  │     └── YES → cloud_only
  │
  └── FAIL → Error: No viable strategy
```

### Task Category Mapping

| Task | Local Model (T4) | Cloud Fallback |
|------|------------------|----------------|
| intent_parsing | Llama-3.2-3B | DeepSeek-V3 |
| code_generation | Qwen-Coder-7B | DeepSeek-V3 |
| code_validation | Qwen-Coder-1.5B | DeepSeek-V3 |
| data_analysis | Phi-3.5-mini | DeepSeek-V3 |
| math_statistics | Qwen-Math-7B | DeepSeek-V3 |
| biomedical | BioMistral-7B | Claude-3.5 |
| documentation | Gemma-2-9B | Claude-3.5 |
| embeddings | BGE-M3 | OpenAI |
| safety | Llama-Guard-3 | Claude-3.5 |
| orchestration | (cloud only) | DeepSeek-V3 |
