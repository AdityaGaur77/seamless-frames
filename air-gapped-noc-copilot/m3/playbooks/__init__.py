"""m3.playbooks: scenario validation playbooks and runner for Phase 6."""
from .fault_injection_runner import SCENARIOS, run_scenario, ScenarioReport
from .validation_metrics import summarise, render_markdown
