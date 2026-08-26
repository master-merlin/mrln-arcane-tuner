# 📡 Observability & Logging Standards

<system_directives>
You are an autonomous AI Agent operating a hybrid FastAPI Web + PyTorch LoRA architecture on a **Windows host**. 
Logs are the application's "Eyes." You must strictly output machine-readable JSON logs for production code. For debugging, you must parse these logs deterministically without exceeding your context window.
</system_directives>

<the_golden_rules>
1. **[STRUCTURED JSON ONLY]** Text-based logs are for humans; JSON logs are for machines. All production application logs MUST be emitted as single-line JSON objects.
2. **[LIBRARIES]** Use `structlog` (preferred) or Python's standard `logging` with a JSON formatter. Angular MUST use a structured custom `LoggerService`.
3. **[THE PRINT BOUNDARY]** 
   - 🚫 `print()` and `console.log()` statements are STRICTLY PROHIBITED in core production code, FastAPI routers, and PyTorch worker scripts.
   - ✅ `print()` is ONLY allowed inside the temporary `tests/adhoc_test_*.py` scripts (as defined in `AGENTS.md`) for quick tensor shape validation.
   - ⚠️ **Bootstrap & safety-net exception.** A small number of `print(..., file=sys.stderr)` calls are permitted in `run_trainer.py` (before `JobLogWriter` is initialised, or as last-resort fallbacks in exception handlers where the logger itself can't be trusted) and in `JobLogWriter` (the double-exit guard). Every such site MUST carry an inline comment of the form `# safety-net print: <reason>`. New safety-net prints require a code-review justification — they are not a general-purpose escape hatch.
</the_golden_rules>

<ml_safety_guardrails>
**CRITICAL HAZARDS FOR PYTORCH LOGGING:**
1. **[THE TENSOR BOMB]** NEVER pass a raw PyTorch tensor into a log statement. It will fail JSON serialization and crash the app. 
   - Extract the scalar: use `loss.item()` instead of `loss`. 
   - Log the shape as a string: `str(tensor.shape)`.
2. **[THE INNER-LOOP FLOOD]** Do NOT emit JSON logs during every step of the inner training loop. Log JSON metrics ONLY at the end of an `epoch`. For high-frequency step metrics, route them to a dedicated ML tracker (e.g., TensorBoard).
</ml_safety_guardrails>

<agent_log_parsing_sop>
**HOW TO READ LOGS ON WINDOWS WITHOUT HANGING:**
If you need to trace a bug, NEVER attempt to read the entire `server.log` file, and NEVER spawn nested `powershell` sub-shells (they will hang waiting for input).
Use your Python environment to safely read and filter the logs deterministically:

1. **Extract by Trace ID:** 
   *Command:* `cmd /c venv\Scripts\python -c "lines = [l for l in open('backend/server.log', encoding='utf-8', errors='ignore') if '<trace_id_here>' in l]; open('.agent/workdir/agent_log_trace.txt', 'w', encoding='utf-8').writelines(lines[-50:])"` 

2. **Extract by Errors:** 
   *Command:* `cmd /c venv\Scripts\python -c "lines = [l for l in open('backend/server.log', encoding='utf-8', errors='ignore') if '\"level\":\"ERROR\"' in l]; open('.agent/workdir/agent_log_errors.txt', 'w', encoding='utf-8').writelines(lines[-20:])"`

3. **Analyze:** Read the output file from `.agent\workdir`, analyze the JSON payloads, and then DELETE it.
</agent_log_parsing_sop>

<universal_json_schema>
Every log entry MUST contain these standard fields to ensure seamless cross-boundary tracing:

```json
{
  "timestamp": "ISO-8601 UTC",
  "level": "INFO|WARNING|ERROR|DEBUG|CRITICAL",
  "service": "fastapi-router|lora-worker|angular-ui",
  "message": "Human and LLM readable summary",
  
  "trace_id": "<UUID from Frontend header>",
  "span_id": "<current operation ID>",
  
  "context": {
    "user_id": 123,
    "vram_allocated_mb": 8450,
    "tensor_shape": "[1, 4, 1024, 1024]",
    "epoch": 5
  }
}
```
</universal_json_schema>