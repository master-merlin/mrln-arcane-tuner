"""Standard training plugin.

Launches training as a subprocess via ``run_trainer.py``, passing the
configuration as JSON.  Also enriches the UI schema with available
dataset names and optimizer/scheduler options.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import structlog
from pydantic import BaseModel

from app.engine.models.base import TrainingPlugin, BaseTrainingConfig

logger = structlog.get_logger(__name__)

class Config(BaseTrainingConfig):
    """Standard training config with a required model definition ID."""

    model_config = {"title": "Config"}


class StandardPlugin(TrainingPlugin):
    """Subprocess-based training backend using ``run_trainer.py``."""

    def get_model_id(self) -> str:
        return "standard"

    def get_config_schema(self) -> type[BaseModel]:
        return Config

    def start_training(self, config: dict[str, any]) -> subprocess.Popen:
        """Launch ``run_trainer.py`` as a subprocess with JSON config."""
        # Resolve paths
        # backend/app/engine/models/standard.py
        current_file = os.path.abspath(__file__)
        # up 4 levels: models -> engine -> app -> backend
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        
        logger.debug("backend_root_resolved", path=backend_root)
        
        # Absolute path to venv python
        python_executable = os.path.abspath(os.path.join(backend_root, "venv", "Scripts", "python.exe"))
        logger.debug("python_executable_path", path=python_executable)
        
        # Fallback for flexibility
        if not os.path.exists(python_executable):
            logger.warning("venv_python_not_found_falling_back", fallback=sys.executable)
            python_executable = sys.executable
            
        # Absolute path to run_trainer.py (it lives in backend root now)
        script_path = os.path.abspath(os.path.join(backend_root, "run_trainer.py"))
        logger.debug("script_path_resolved", path=script_path)
        
        definition_id = config.get("definition_id")
        if not definition_id:
            definition_id = "sdxl_base_1.0"
            
        cmd = [
            python_executable,
            "-u",
            script_path,
            "--definition_id", definition_id,
            "--config", json.dumps(config),
        ]
        
        logger.info("launching_trainer", command=" ".join(cmd))
        
        # Detach the subprocess so it survives backend restarts.
        # On Windows, DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP prevents
        # the child from receiving console signals (CTRL+C, etc.) and from
        # being killed when the parent's console closes.
        # stdout/stderr → DEVNULL: the trainer writes to job_log.jsonl via
        # JobLogWriter instead of stdout pipes, which break on parent exit.
        creation_flags = 0
        if os.name == "nt":
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )

        try:
            from pathlib import Path
            output_dir = config.get("output_dir")
            job_id = config.get("job_id", "unknown")
            
            if output_dir:
                boot_log_path = Path(output_dir) / "trainer_stdout.log"
            else:
                # Fallback purely if config lacks output_dir
                boot_log_path = Path(backend_root) / "data" / "outputs" / f"boot_{job_id}.log"
                
            boot_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Use line buffering to ensure tracebacks are immediately written
            boot_log_file = open(boot_log_path, "w", buffering=1)
        except Exception:
            boot_log_file = subprocess.DEVNULL

        process = subprocess.Popen(
            cmd,
            stdout=boot_log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=os.environ.copy(),
            cwd=backend_root,
            creationflags=creation_flags,
            start_new_session=(os.name != "nt"),
        )
        
        return process

    def enrich_schema(self, schema: dict[str, any]) -> dict[str, any]:
        """Inject available datasets and optimizer options into the UI schema."""
        schema = super().enrich_schema(schema)

        from app.core.dataset_manager import dataset_manager
        dataset_names = [ds.name for ds in dataset_manager.list_datasets()]
        
        try:
            # Check properties directly
            if "properties" in schema and "datasets" in schema["properties"]:
                 items = schema["properties"]["datasets"].get("items", {})
                 if "properties" in items and "dataset_name" in items["properties"]:
                     items["properties"]["dataset_name"]["enum"] = dataset_names
            
            # Check $defs
            defs = schema.get("$defs", {})
            for def_name, def_val in defs.items():
                if "properties" in def_val and "dataset_name" in def_val["properties"]:
                    def_val["properties"]["dataset_name"]["enum"] = dataset_names
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("schema_enrichment_failed", error=str(e))
            
        # Enrich Optimizer & Scheduler options
        try:
            from app.engine.factories.optimizer import OptimizerFactory, LRSchedulerFactory
            supported_optimizers = list(OptimizerFactory.SUPPORTED_OPTIMIZERS.keys())
            supported_schedulers = LRSchedulerFactory.SUPPORTED_SCHEDULERS
            
            if "properties" in schema:
                if "optimizer_type" in schema["properties"]:
                    schema["properties"]["optimizer_type"]["enum"] = supported_optimizers
                if "lr_scheduler" in schema["properties"]:
                    schema["properties"]["lr_scheduler"]["enum"] = supported_schedulers
                    
        except (ImportError, KeyError, TypeError, AttributeError) as e:
            logger.error("schema_enrichment_optimizers_failed", error=str(e))
            
        return schema
