class JobOutputFields:
    GENERATED_GOLDEN_DATASET = "generated_golden_dataset_path"
    COMPARE_MODELS_INFERENCE_OUTPUT = "compare_models_inference_output_path"
    EVALUATING_INFERENCE_OUTPUT = "evaluating_inference_output_path"
    SCALE_AND_COST_PROJECTION_REPORT = "scale_and_cost_projection_report_path"


class JobStatusStr:
    QUEUED = "queued"
    VALIDATE_INPUT = "validate_input"
    DETECTING_WORKLOAD_TYPE = "detecting_workload_type"
    GENERATE_GOLDEN_DATASET = "generate_golden_dataset"
    RUNNING = "running"
    RUN_COMPARE_MODELS_INFERENCE = "run_compare_models_inference"
    EVALUATING_INFERENCE_OUTPUT = "run_compare_models_evaluation"
    CREATE_SCALE_AND_COST_PROJECTION_REPORT = "create_scale_and_cost_projection_report"
    CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML = "create_scale_and_cost_projection_report_html"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


RUNNING_STATUSES: frozenset[str] = frozenset(
    {
        JobStatusStr.QUEUED,
        JobStatusStr.VALIDATE_INPUT,
        JobStatusStr.DETECTING_WORKLOAD_TYPE,
        JobStatusStr.GENERATE_GOLDEN_DATASET,
        JobStatusStr.RUNNING,
        JobStatusStr.EVALUATING_INFERENCE_OUTPUT,
        JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
    }
)

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        JobStatusStr.COMPLETED,
        JobStatusStr.FAILED,
        JobStatusStr.CANCELLED,
    }
)

INFERENCE_BLOB_FIELDS: list[str] = [
    JobOutputFields.COMPARE_MODELS_INFERENCE_OUTPUT,
    JobOutputFields.EVALUATING_INFERENCE_OUTPUT,
    JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT,
]

PIPELINE_STEPS: list[str] = [
    JobStatusStr.GENERATE_GOLDEN_DATASET,
    JobStatusStr.RUN_COMPARE_MODELS_INFERENCE,
    JobStatusStr.EVALUATING_INFERENCE_OUTPUT,
    JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
    JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML,
]
