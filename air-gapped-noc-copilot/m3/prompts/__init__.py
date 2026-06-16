"""m3.prompts: system prompt, response schema, validator, and assembler for the offline LLM Copilot."""
from .prompt_assembler import assemble, load_system_prompt, format_evidence_block, format_alert_block, format_question_block
from .schema_validator import validate_response, CopilotUnavailable
