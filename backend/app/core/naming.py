"""Shared run-name / output-dir naming helpers.

`model_part_from_definition_id` MUST remain the single source of truth. It was
previously copy-pasted across run_trainer, job_routes, job_manager,
pipeline_optimization and sampling, where any drift would silently send a run's
output to a different directory than the trainer expected.
"""


def model_part_from_definition_id(definition_id: str) -> str:
    """Derive the filesystem-safe model segment of a run name from a definition id.

    Drops the org/namespace prefix (``ostris/flux-dev`` -> ``flux-dev``) and
    replaces ``:`` (e.g. a turbo variant suffix) with ``_`` so the result is a
    valid path segment on Windows.
    """
    return definition_id.split("/")[-1].replace(":", "_")
