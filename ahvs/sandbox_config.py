"""Standalone sandbox and experiment configuration dataclasses.

Extracted from the full ResearchClaw config for use by BB_AHVS.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxConfig:
    python_path: str = ".venv/bin/python3"
    gpu_required: bool = False
    allowed_imports: tuple[str, ...] = (
        "math",
        "random",
        "json",
        "csv",
        "numpy",
        "torch",
        "sklearn",
    )
    max_memory_mb: int = 4096


@dataclass(frozen=True)
class SshRemoteConfig:
    host: str = ""
    user: str = ""
    port: int = 22
    key_path: str = ""
    gpu_ids: tuple[int, ...] = ()
    remote_workdir: str = "/tmp/researchclaw_experiments"
    remote_python: str = "python3"
    setup_commands: tuple[str, ...] = ()
    use_docker: bool = False
    docker_image: str = "researchclaw/experiment:latest"
    docker_network_policy: str = "none"
    docker_memory_limit_mb: int = 8192
    docker_shm_size_mb: int = 2048


@dataclass(frozen=True)
class ColabDriveConfig:
    """Configuration for Google Drive-based async Colab execution."""

    drive_root: str = ""  # local mount path, e.g. ~/Google Drive/MyDrive/researchclaw
    poll_interval_sec: int = 30
    timeout_sec: int = 3600
    setup_script: str = ""  # commands to run before experiment, written to setup.sh


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Configuration for Docker-based experiment sandbox."""

    image: str = "researchclaw/experiment:latest"
    gpu_enabled: bool = True
    gpu_device_ids: tuple[int, ...] = ()
    memory_limit_mb: int = 8192
    network_policy: str = "setup_only"  # none | setup_only | pip_only | full
    pip_pre_install: tuple[str, ...] = ()
    auto_install_deps: bool = True
    shm_size_mb: int = 2048
    container_python: str = "/usr/bin/python3"
    keep_containers: bool = False


@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for the advanced multi-phase code generation agent."""

    enabled: bool = True
    # Phase 1: Blueprint planning (deep implementation blueprint)
    architecture_planning: bool = True
    # Phase 2: Sequential file generation (one-by-one following blueprint)
    sequential_generation: bool = True
    # Phase 2.5: Hard validation gates (AST-based)
    hard_validation: bool = True
    hard_validation_max_repairs: int = 2
    # Phase 3: Execution-in-the-loop (run -> parse error -> fix)
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60
    # Phase 4: Solution tree search (off by default — higher cost)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2
    tree_search_eval_timeout_sec: int = 120
    # Phase 5: Multi-agent review dialog
    review_max_rounds: int = 2


@dataclass(frozen=True)
class BenchmarkAgentConfig:
    """Configuration for the BenchmarkAgent multi-agent system."""

    enabled: bool = True
    # Surveyor
    enable_hf_search: bool = True
    max_hf_results: int = 10
    # Selector
    tier_limit: int = 2
    min_benchmarks: int = 1
    min_baselines: int = 2
    prefer_cached: bool = True
    # Orchestrator
    max_iterations: int = 2


@dataclass(frozen=True)
class FigureAgentConfig:
    """Configuration for the FigureAgent multi-agent system."""

    enabled: bool = True
    # Planner
    min_figures: int = 3
    max_figures: int = 8
    # Orchestrator
    max_iterations: int = 3  # max CodeGen->Renderer->Critic retry loops
    # Renderer security
    render_timeout_sec: int = 30
    use_docker: bool | None = None  # None = auto-detect, True/False to force
    docker_image: str = "researchclaw/experiment:latest"
    # Code generation output format
    output_format: str = "python"  # "python" (matplotlib) or "latex" (TikZ/PGFPlots)
    # Nano Banana (Gemini image generation)
    gemini_api_key: str = ""  # or set GEMINI_API_KEY / GOOGLE_API_KEY env var
    gemini_model: str = "gemini-2.5-flash-image"
    nano_banana_enabled: bool = True  # enable/disable Gemini image generation
    # Critic
    strict_mode: bool = False
    # Output
    dpi: int = 300


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "simulated"
    time_budget_sec: int = 300
    max_iterations: int = 10
    metric_key: str = "primary_metric"
    metric_direction: str = "minimize"
    keep_threshold: float = 0.0
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    docker: DockerSandboxConfig = field(default_factory=DockerSandboxConfig)
    ssh_remote: SshRemoteConfig = field(default_factory=SshRemoteConfig)
    colab_drive: ColabDriveConfig = field(default_factory=ColabDriveConfig)
    code_agent: CodeAgentConfig = field(default_factory=CodeAgentConfig)
    benchmark_agent: BenchmarkAgentConfig = field(default_factory=BenchmarkAgentConfig)
    figure_agent: FigureAgentConfig = field(default_factory=FigureAgentConfig)
